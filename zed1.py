########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS



# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import cv2
import numpy as np


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
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    body_params = sl.BodyTrackingParameters()
    # Different model can be chosen, optimizing the runtime or the accuracy
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    # Optimize the person joints position, requires more computations
    body_params.enable_body_fitting = True

    if body_params.enable_tracking:
        positional_tracking_param = sl.PositionalTrackingParameters()
        # positional_tracking_param.set_as_static = True
        positional_tracking_param.set_floor_as_origin = True
        zed.enable_positional_tracking(positional_tracking_param)

    print("Body tracking: Loading Module...")

    err = zed.enable_body_tracking(body_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print("Enable Body Tracking : "+repr(err)+". Exit program.")
        zed.close()
        exit()
    bodies = sl.Bodies()
    image_zed=sl.Mat()
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    # For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
    # For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
    body_runtime_param.detection_confidence_threshold = 40

    while (True):
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            zed.retrieve_image(image_zed,sl.VIEW.LEFT)
            image=image_zed.get_data()
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            cv2.putText(
                image,
                "ZED Body Tracking View",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if bodies.is_new:
                body_array = bodies.body_list
                print(str(len(body_array)) + " Person(s) detected\n")

                cv2.putText(
                    image,
                    f"Persons detected: {len(body_array)}",
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                if len(body_array) > 0:
                    first_body = body_array[0]
                    print("First Person attributes:")
                    print(" Confidence (" + str(int(first_body.confidence)) + "/100)")
                    if body_params.enable_tracking:
                        print(" Tracking ID: " + str(int(first_body.id)) + " tracking state: " + repr(
                            first_body.tracking_state) + " / " + repr(first_body.action_state))
                    position = first_body.position
                    velocity = first_body.velocity
                    dimensions = first_body.dimensions
                    
                    cv2.putText(image,f"id:{first_body.id}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

                    cv2.putText(
                        image,
                        f"Pos: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        f"Vel: vx={velocity[0]:.2f}, vy={velocity[1]:.2f}, vz={velocity[2]:.2f}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        f"Confidence: {int(first_body.confidence)}/100",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        f"Tracking: {first_body.tracking_state}",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        f"Action: {first_body.action_state}",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )


                    print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(
                        position[0], position[1], position[2], velocity[0], velocity[1], velocity[2], dimensions[0],
                        dimensions[1], dimensions[2]))
                    if first_body.mask.is_init():
                        print(" 2D mask available")

                    print(" Keypoint 2D ")
                    keypoint_2d = first_body.keypoint_2d
                    for it in keypoint_2d:
                        print("    " + str(it))

                    skeleton_pairs = [
                        (0, 1),
                        (1, 2), (2, 3), (3, 4),
                        (1, 5), (5, 6), (6, 7),
                        (1, 8), (8, 9), (9, 10),
                        (1, 11), (11, 12), (12, 13),
                        (0, 14), (14, 16),
                        (0, 15), (15, 17)
                    ]

                    for p1, p2 in skeleton_pairs:
                        if p1 < len(keypoint_2d) and p2 < len(keypoint_2d):
                            pt1 = keypoint_2d[p1]
                            pt2 = keypoint_2d[p2]

                            x1, y1 = pt1[0], pt1[1]
                            x2, y2 = pt2[0], pt2[1]

                            if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
                                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                    cv2.line(
                                        image,
                                        (int(x1), int(y1)),
                                        (int(x2), int(y2)),
                                        (0, 255, 0),
                                        2
                                    )

                    for idx, pt in enumerate(keypoint_2d):
                        x, y = pt[0], pt[1]

                        if not np.isnan(x) and not np.isnan(y):
                            if x > 0 and y > 0:
                                cv2.circle(
                                    image,
                                    (int(x), int(y)),
                                    5,
                                    (0, 0, 255),
                                    -1
                                )

                                cv2.putText(
                                    image,
                                    str(idx),
                                    (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (255, 255, 255),
                                    1
                                )

                    print("\n Keypoint 3D ")
                    keypoint = first_body.keypoint
                    for it in keypoint:
                        print("    " + str(it))

                    for idx, pt2d in enumerate(keypoint_2d):
                        x2d, y2d = pt2d[0], pt2d[1]

                        if not np.isnan(x2d) and not np.isnan(y2d):
                            if x2d > 0 and y2d > 0 and idx < len(keypoint):
                                kp3d = keypoint[idx]

                                cv2.putText(
                                    image,
                                    f"({kp3d[0]:.1f},{kp3d[1]:.1f},{kp3d[2]:.1f})",
                                    (int(x2d) + 8, int(y2d) + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (0, 255, 0),
                                    1
                                )
        

        cv2.imshow("zed camera view",image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Close the camera
    zed.disable_body_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()