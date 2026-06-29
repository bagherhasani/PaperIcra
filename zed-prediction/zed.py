########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#
########################################################################

import pyzed.sl as sl
import cv2
import numpy as np
from kf_zed import KF


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    body_params = sl.BodyTrackingParameters()
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    body_params.enable_body_fitting = True

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

    # KF variables
    kf = None
    previous_timestamp = None
    seconds_ahead = 1.0

    # Trajectory UI variables
    trajectory_points = []
    future_points = []
    max_trajectory_len = 25

    while True:
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            err = zed.retrieve_bodies(bodies, body_runtime_param)

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image = image_zed.get_data()

            # Better ZED color conversion
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            if bodies.is_new:
                body_array = bodies.body_list
                print(str(len(body_array)) + " Person(s) detected\n")

                if len(body_array) > 0:
                    first_body = body_array[0]

                    print("First Person attributes:")
                    print(" Confidence (" + str(int(first_body.confidence)) + "/100)")

                    if body_params.enable_tracking:
                        print(
                            " Tracking ID: " + str(int(first_body.id)) +
                            " tracking state: " + repr(first_body.tracking_state) +
                            " / " + repr(first_body.action_state)
                        )

                    position = first_body.position
                    velocity = first_body.velocity
                    dimensions = first_body.dimensions

                    print(
                        " 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(
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

                    if first_body.mask.is_init():
                        print(" 2D mask available")

                    print(" Keypoint 2D ")
                    keypoint_2d = first_body.keypoint_2d
                    for it in keypoint_2d:
                        print("    " + str(it))

                    print("\n Keypoint 3D ")
                    keypoint = first_body.keypoint
                    for it in keypoint:
                        print("    " + str(it))

                    # -----------------------------
                    # KF measurement
                    # -----------------------------
                    measured_px = position[2]   # forward/backward
                    measured_py = position[0]   # left/right

                    current_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()

                    if previous_timestamp is None:
                        dt = 0.033
                    else:
                        dt = current_timestamp - previous_timestamp

                    previous_timestamp = current_timestamp

                    if kf is None:
                        kf = KF(
                            measured_px,
                            measured_py,
                            0.0,
                            0.0,
                            dt
                        )

                    px, py, vx, vy = kf.processMeasurement(
                        measured_px,
                        measured_py,
                        dt
                    )

                    future_px, future_py = kf.predictFuture(seconds_ahead)

                    # -----------------------------
                    # Simple UI: rectangle + trajectory + vector
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

                            # rectangle around person
                            cv2.rectangle(
                                image,
                                (x_min, y_min),
                                (x_max, y_max),
                                (0, 255, 0),
                                3
                            )

                            # person center
                            cv2.circle(
                                image,
                                (center_x, center_y),
                                6,
                                (0, 0, 255),
                                -1
                            )

                            # -----------------------------
                            # Current trajectory trail
                            # -----------------------------
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
                            # Prediction vector
                            # -----------------------------
                            dx = future_py - py      # lateral motion
                            dy = future_px - px      # forward motion

                            arrow_scale = 700

                            arrow_end_x = int(center_x + dx * arrow_scale)
                            arrow_end_y = int(center_y - dy * arrow_scale)

                            # Save future predicted point history
                            future_points.append((arrow_end_x, arrow_end_y))

                            if len(future_points) > max_trajectory_len:
                                future_points.pop(0)

                            # Draw future prediction trajectory trail
                            if len(future_points) > 1:
                                cv2.polylines(
                                    image,
                                    [np.array(future_points, dtype=np.int32)],
                                    False,
                                    (255, 0, 255),
                                    3
                                )

                            # Draw current prediction arrow
                            cv2.arrowedLine(
                                image,
                                (center_x, center_y),
                                (arrow_end_x, arrow_end_y),
                                (255, 0, 255),
                                5,
                                tipLength=0.35
                            )

                            cv2.circle(
                                image,
                                (arrow_end_x, arrow_end_y),
                                8,
                                (255, 0, 255),
                                -1
                            )

                    # -----------------------------
                    # Simple text only
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
                        f"KF: forward={px:.2f}, lateral={py:.2f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        f"Future {seconds_ahead}s: forward={future_px:.2f}, lateral={future_py:.2f}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        "Yellow = current trajectory | Purple = prediction trajectory",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

            cv2.imshow("zed camera view", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()