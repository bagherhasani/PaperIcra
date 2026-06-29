########################################################################
#
# ZED BODY_18 Hip Heading + EKF Pedestrian Prediction
#
########################################################################

import pyzed.sl as sl
import cv2
import numpy as np
from ekf_zed import Ekf


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_velocity_heading(velocity):
    """
    Fallback heading from ZED velocity.

    EKF coordinates:
        forward = velocity[2]
        lateral = velocity[0]
    """

    vx_forward = velocity[2]
    vy_lateral = velocity[0]

    speed = np.sqrt(
        vx_forward * vx_forward +
        vy_lateral * vy_lateral
    )

    if speed < 0.05:
        return None

    heading = np.arctan2(vy_lateral, vx_forward)
    heading = normalize_angle(heading)

    return heading


def get_hip_heading_body18(keypoint_3d, velocity):
    """
    BODY_18 hip heading.

    BODY_18 indices:
        right hip = 8
        left hip  = 11

    ZED coordinates:
        x = lateral left/right
        y = vertical
        z = forward/backward

    EKF coordinates:
        px = z
        py = x
    """

    if keypoint_3d is None or len(keypoint_3d) < 12:
        return get_velocity_heading(velocity)

    right_hip = keypoint_3d[8]
    left_hip = keypoint_3d[11]

    if np.any(np.isnan(right_hip)) or np.any(np.isnan(left_hip)):
        return get_velocity_heading(velocity)

    # Hip side vector across pelvis
    hip_side_x = right_hip[0] - left_hip[0]
    hip_side_z = right_hip[2] - left_hip[2]

    norm = np.sqrt(
        hip_side_x * hip_side_x +
        hip_side_z * hip_side_z
    )

    if norm < 1e-6:
        return get_velocity_heading(velocity)

    hip_side_x = hip_side_x / norm
    hip_side_z = hip_side_z / norm

    # Body forward direction is perpendicular to hip line
    hip_forward_x = -hip_side_z
    hip_forward_z = hip_side_x

    # Fix front/back ambiguity using velocity
    vel_x = velocity[0]
    vel_z = velocity[2]

    vel_norm = np.sqrt(
        vel_x * vel_x +
        vel_z * vel_z
    )

    if vel_norm > 0.05:
        vel_x = vel_x / vel_norm
        vel_z = vel_z / vel_norm

        dot = hip_forward_x * vel_x + hip_forward_z * vel_z

        if dot < 0:
            hip_forward_x = -hip_forward_x
            hip_forward_z = -hip_forward_z

    hip_lateral = hip_forward_x
    hip_forward = hip_forward_z

    hip_heading = np.arctan2(hip_lateral, hip_forward)
    hip_heading = normalize_angle(hip_heading)

    return hip_heading


def draw_top_down_prediction(image, px, py, future_trajectory):
    """
    Draw EKF prediction on a top-down map.

    This fixes the problem where the prediction arrow goes to the ceiling
    when the person is far from the camera.
    """

    map_w = 300
    map_h = 300
    scale = 60  # pixels per meter

    x0 = 20
    y0 = image.shape[0] - map_h - 20

    # Background
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + map_w, y0 + map_h),
        (30, 30, 30),
        -1
    )

    # Border
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + map_w, y0 + map_h),
        (255, 255, 255),
        2
    )

    # Camera location
    cam_x = x0 + map_w // 2
    cam_y = y0 + map_h - 30

    cv2.circle(image, (cam_x, cam_y), 6, (255, 255, 255), -1)

    cv2.putText(
        image,
        "Camera",
        (cam_x - 35, cam_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    def ground_to_pixel(forward, lateral):
        pixel_x = int(cam_x + lateral * scale)
        pixel_y = int(cam_y - forward * scale)
        return pixel_x, pixel_y

    # Current EKF position
    current_point = ground_to_pixel(px, py)

    cv2.circle(
        image,
        current_point,
        6,
        (0, 255, 255),
        -1
    )

    # Future trajectory
    future_pixels = []

    for f_px, f_py in future_trajectory:
        future_pixels.append(
            ground_to_pixel(f_px, f_py)
        )

    if len(future_pixels) > 1:
        cv2.polylines(
            image,
            [np.array(future_pixels, dtype=np.int32)],
            False,
            (255, 0, 255),
            3
        )

    if len(future_pixels) > 0:
        cv2.circle(
            image,
            future_pixels[-1],
            6,
            (255, 0, 255),
            -1
        )

    cv2.putText(
        image,
        "Top-down EKF prediction",
        (x0 + 10, y0 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# MAIN ZED CODE
# ============================================================

def main():
    # Create camera object
    zed = sl.Camera()

    # Camera initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open camera
    err = zed.open(init_params)

    if err > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    # Body tracking parameters
    body_params = sl.BodyTrackingParameters()
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    body_params.enable_body_fitting = True

    # Use BODY_18 if available
    try:
        body_params.body_format = sl.BODY_FORMAT.BODY_18
    except AttributeError:
        print("BODY_18 format setting not available in this SDK version.")
        print("Continuing with default body format.")

    # Positional tracking
    if body_params.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Body tracking: Loading Module...")

    err = zed.enable_body_tracking(body_params)

    if err > sl.ERROR_CODE.SUCCESS:
        print("Enable Body Tracking : " + repr(err) + ". Exit program.")
        zed.close()
        exit()

    bodies = sl.Bodies()
    image_zed = sl.Mat()

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # EKF variables
    ekf = None
    previous_timestamp = None
    seconds_ahead = 2.0

    # Evaluation variables
    prediction_buffer = []
    prediction_errors = []

    csv_file = open("ekf_prediction_log.csv", "w")
    csv_file.write(
        "time,actual_px,actual_py,pred_px_1s,pred_py_1s,error_1s,speed,heading,heading_rate\n"
    )

    # UI variables
    trajectory_points = []
    max_trajectory_len = 25

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_bodies(bodies, body_runtime_param)

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image = image_zed.get_data()

            # Convert ZED BGRA image to OpenCV BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            if bodies.is_new:
                body_array = bodies.body_list

                if len(body_array) > 0:
                    first_body = body_array[0]

                    position = first_body.position
                    velocity = first_body.velocity
                    dimensions = first_body.dimensions
                    keypoint = first_body.keypoint

                    print(str(len(body_array)) + " Person(s) detected\n")

                    print("First Person attributes:")
                    print(" Confidence (" + str(int(first_body.confidence)) + "/100)")

                    if body_params.enable_tracking:
                        print(
                            " Tracking ID: " + str(int(first_body.id)) +
                            " tracking state: " + repr(first_body.tracking_state) +
                            " / " + repr(first_body.action_state)
                        )

                    print(
                        " 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimensions: [{6},{7},{8}]".format(
                            position[0],
                            position[1],
                            position[2],
                            velocity[0],
                            velocity[1],
                            velocity[2],
                            dimensions[0],
                            dimensions[1],
                            dimensions[2]
                        )
                    )

                    # -----------------------------
                    # EKF measurement
                    # -----------------------------

                    measured_px = position[2]   # forward/backward
                    measured_py = position[0]   # left/right

                    measured_heading = get_hip_heading_body18(
                        keypoint,
                        velocity
                    )

                    current_timestamp = zed.get_timestamp(
                        sl.TIME_REFERENCE.IMAGE
                    ).get_seconds()

                    if previous_timestamp is None:
                        dt = 0.033
                    else:
                        dt = current_timestamp - previous_timestamp

                    previous_timestamp = current_timestamp

                    # Safety for weird dt
                    if dt <= 0 or dt > 1.0:
                        dt = 0.033

                    # Only run EKF if heading is available
                    if measured_heading is not None:

                        zed_v_forward = velocity[2]
                        zed_v_lateral = velocity[0]

                        initial_speed = np.sqrt(
                            zed_v_forward * zed_v_forward +
                            zed_v_lateral * zed_v_lateral
                        )

                        if ekf is None:
                            ekf = Ekf(
                                measured_px,
                                measured_py,
                                initial_speed,
                                measured_heading,
                                0.0,
                                dt
                            )

                        px, py, speed, heading, heading_rate = ekf.process_measurement(
                            measured_px,
                            measured_py,
                            measured_heading,
                            dt
                        )

                        future_px, future_py = ekf.predictFuture(seconds_ahead)

                        future_trajectory = ekf.predictFutureTrajectory(
                            seconds_ahead,
                            steps=20
                        )

                        # -----------------------------
                        # Store prediction for later evaluation
                        # -----------------------------

                        prediction_buffer.append({
                            "target_time": current_timestamp + seconds_ahead,
                            "pred_px": future_px,
                            "pred_py": future_py,
                            "speed": speed,
                            "heading": heading,
                            "heading_rate": heading_rate
                        })

                        # -----------------------------
                        # Evaluate old predictions
                        # -----------------------------

                        remaining_predictions = []

                        for pred in prediction_buffer:
                            if current_timestamp >= pred["target_time"]:

                                actual_px_now = measured_px
                                actual_py_now = measured_py

                                error = np.sqrt(
                                    (pred["pred_px"] - actual_px_now) ** 2 +
                                    (pred["pred_py"] - actual_py_now) ** 2
                                )

                                prediction_errors.append(error)

                                csv_file.write(
                                    f"{current_timestamp},"
                                    f"{actual_px_now},"
                                    f"{actual_py_now},"
                                    f"{pred['pred_px']},"
                                    f"{pred['pred_py']},"
                                    f"{error},"
                                    f"{pred['speed']},"
                                    f"{pred['heading']},"
                                    f"{pred['heading_rate']}\n"
                                )

                                csv_file.flush()

                                ade = np.mean(prediction_errors)

                                print(f"1s prediction error = {error:.3f} m")
                                print(f"ADE so far = {ade:.3f} m")

                            else:
                                remaining_predictions.append(pred)

                        prediction_buffer = remaining_predictions

                        # -----------------------------
                        # UI: rectangle + current trajectory
                        # -----------------------------

                        bbox = first_body.bounding_box_2d

                        if bbox is not None and len(bbox) >= 4:
                            xs = [p[0] for p in bbox]
                            ys = [p[1] for p in bbox]

                            if not any(np.isnan(xs)) and not any(np.isnan(ys)):
                                x_min = int(min(xs))
                                x_max = int(max(xs))
                                y_min = int(min(ys))
                                y_max = int(max(ys))

                                center_x = int((x_min + x_max) / 2)
                                center_y = int((y_min + y_max) / 2)

                                # Rectangle around person
                                cv2.rectangle(
                                    image,
                                    (x_min, y_min),
                                    (x_max, y_max),
                                    (0, 255, 0),
                                    3
                                )

                                # Person center
                                cv2.circle(
                                    image,
                                    (center_x, center_y),
                                    6,
                                    (0, 0, 255),
                                    -1
                                )

                                # Current camera-image trajectory trail
                                trajectory_points.append((center_x, center_y))

                                if len(trajectory_points) > max_trajectory_len:
                                    trajectory_points.pop(0)

                                if len(trajectory_points) > 1:
                                    cv2.polylines(
                                        image,
                                        [np.array(trajectory_points, dtype=np.int32)],
                                        False,
                                        (0, 255, 255),
                                        3
                                    )

                        # -----------------------------
                        # Top-down EKF prediction map
                        # -----------------------------

                        draw_top_down_prediction(
                            image,
                            px,
                            py,
                            future_trajectory
                        )

                        # -----------------------------
                        # Text UI
                        # -----------------------------

                        cv2.putText(
                            image,
                            f"ID:{first_body.id}  Conf:{int(first_body.confidence)}%",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            image,
                            f"ZED: x={position[0]:.2f}, z={position[2]:.2f}",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            image,
                            f"EKF: forward={px:.2f}, lateral={py:.2f}, heading={heading:.2f}",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 255),
                            2
                        )

                        cv2.putText(
                            image,
                            f"Speed={speed:.2f}, heading_rate={heading_rate:.2f}",
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 255),
                            2
                        )

                        cv2.putText(
                            image,
                            f"Future {seconds_ahead}s: forward={future_px:.2f}, lateral={future_py:.2f}",
                            (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 255),
                            2
                        )

                        if len(prediction_errors) > 0:
                            ade = np.mean(prediction_errors)

                            cv2.putText(
                                image,
                                f"1s ADE so far: {ade:.3f} m",
                                (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                        cv2.putText(
                            image,
                            "Yellow = current image path | Purple = top-down EKF prediction",
                            (20, 220),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2
                        )

                    else:
                        cv2.putText(
                            image,
                            "Hip heading not available",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

            cv2.imshow("zed camera view", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    csv_file.close()

    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()