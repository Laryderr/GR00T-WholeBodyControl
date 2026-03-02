import argparse
import csv
import os
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
import numpy as np

from lxml import etree

import zmq
import threading
import msgpack

measured_root_state = None


def _is_motion_dir(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "joint_pos.csv"))
        and os.path.isfile(os.path.join(path, "body_pos.csv"))
        and os.path.isfile(os.path.join(path, "body_quat.csv"))
    )


def key_call_back(keycode):
    global \
        curr_start, \
        num_motions, \
        motion_id, \
        motion_acc, \
        time_step, \
        dt, \
        paused, \
        data_csv_dict, \
        frame_idx, \
        anim_idx, \
        measured_root_state
    
    try:
        c = chr(keycode)
    except:
        c = ""
    if c == "R":
        print("Reset")
        frame_idx = int(0)
        if measured_root_state is not None:
            with measured_root_state["lock"]:
                measured_root_state["need_realign"] = True
                measured_root_state["reported_realign"] = False
    elif c == " ":
        print("Paused")
        paused = not paused
    elif c == ".":
        frame_idx = frame_idx + 1
        print("frame", frame_idx)
    elif c == ",":
        frame_idx = frame_idx - 1
        print("frame", frame_idx)
    elif c == "=":
        anim_idx = anim_idx + 1
        print("anim", anim_idx)
    elif c == "-":
        anim_idx = anim_idx - 1
        print("anim", anim_idx)
    else:
        print("not mapped", c)


def load_anim_data(csv_path: str):

    ret = []
    if os.path.isdir(csv_path):
        # Case A: csv_path itself is a motion directory
        if _is_motion_dir(csv_path):
            motion_dirs = [csv_path]
        else:
            # Case B: csv_path is a parent directory containing multiple motions
            motion_dirs = [
                str(p)
                for p in sorted(Path(csv_path).iterdir())
                if p.is_dir() and _is_motion_dir(str(p))
            ]
            if len(motion_dirs) == 0:
                raise ValueError(
                    f"No motion subdirectories found in {csv_path}. "
                    "Each motion dir must contain joint_pos.csv/body_pos.csv/body_quat.csv."
                )
            print(f"[INFO] Discovered {len(motion_dirs)} motions under {csv_path}")
            for i, md in enumerate(motion_dirs, start=1):
                print(f"  {i:03d}. {md}")

        for motion_dir in motion_dirs:
            joint_pos_path = os.path.join(motion_dir, "joint_pos.csv")
            body_pos_path = os.path.join(motion_dir, "body_pos.csv")
            body_quat_path = os.path.join(motion_dir, "body_quat.csv")

            isaaclab_to_mujoco = [0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
                                11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]

            with open(joint_pos_path, mode="r", newline="") as joint_pos_file, open(body_pos_path, mode="r", newline="") as body_pos_file, open(body_quat_path, mode="r", newline="") as body_quat_file:
                firstRow = True
                joint_pos_rowlist = []
                body_pos_rowlist = []
                body_quat_rowlist = []
                for joint_pos_row, body_pos_row, body_quat_row in zip(joint_pos_file, body_pos_file, body_quat_file):
                    if firstRow:
                        firstRow = False
                        continue

                    joint_pos_row = np.array([float(x) for x in joint_pos_row.split(",")])
                    body_pos_row = np.array([float(x) for x in body_pos_row.split(",")])
                    body_quat_row = np.array([float(x) for x in body_quat_row.split(",")])

                    joint_pos_rowlist.append(joint_pos_row)
                    body_pos_rowlist.append(body_pos_row)
                    body_quat_rowlist.append(body_quat_row)

                ret.append({
                    "dof": np.array(joint_pos_rowlist)[:, isaaclab_to_mujoco],
                    "root_rot": np.array(body_quat_rowlist)[:, [0, 1, 2, 3]],  # [x, y, z, w]
                    "root_trans_offset": np.array(body_pos_rowlist)[:, :3],
                })

    else:
        csv_data = []
        current_rowlist = []
        with open(csv_path, mode="r", newline="") as file:

            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row):
                    r = [x for x in row if x]
                    assert len(r) == 36
                    current_rowlist.append(r)
                else:
                    csv_data.append(current_rowlist)
                    current_rowlist = []

            if current_rowlist:
                csv_data.append(current_rowlist)

        for d in csv_data:
            ret.append({
                "dof": np.array(d)[:, 7:],
                "root_rot": np.array(d)[:, 3:7][:, [0, 1, 2, 3]],  # [x, y, z, w]
                "root_trans_offset": np.array(d)[:, :3],
            })

    return ret


def receive_realtime_debug_messages(socket, data_csv_dicts, topic):
    while True:
        message = socket.recv()

        # Remove any header or leading bytes (should be exactly 8 bytes for "g1_debug")
        data = message.split(topic.encode())[1]

        result = msgpack.unpackb(data)

        data_csv_dicts[0]["root_trans_offset"][0, ...] = result["base_trans_target"]
        data_csv_dicts[0]["root_rot"][0, ...] = result["base_quat_target"]
        data_csv_dicts[0]["dof"][0, ...] = result["body_q_target"]

        data_csv_dicts[0]["root_trans_offset_measured"][0, ...] = result["base_trans_measured"]
        data_csv_dicts[0]["root_rot_measured"][0, ...] = result["base_quat_measured"]
        data_csv_dicts[0]["dof_measured"][0, ...] = result["body_q_measured"]

        data_csv_dicts[0]["vr_3point_position"] = np.array(result["vr_3point_position"]).reshape(3,3)
        data_csv_dicts[0]["vr_3point_orientation"] = np.array(result["vr_3point_orientation"]).reshape(3,4)
        data_csv_dicts[0]["vr_3point_compliance"] = np.array(result["vr_3point_compliance"]).reshape(3)


def create_measured_root_state():
    return {
        "lock": threading.Lock(),
        "has_valid_odom": False,
        "odom_pos": np.zeros(3, dtype=np.float64),
        "odom_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "alignment_offset": np.zeros(3, dtype=np.float64),
        "need_realign": True,
        "reported_realign": False,
        "reported_odom_active": False,
        "subscription_error": "",
        "reported_subscription_error": False,
    }


def has_valid_odometry(shared_state):
    with shared_state["lock"]:
        return shared_state["has_valid_odom"]


def wait_for_odometry(shared_state, timeout_sec):
    start = time.time()
    while time.time() - start < timeout_sec:
        if has_valid_odometry(shared_state):
            return True
        time.sleep(0.02)
    return False


def init_dds_for_odometry(domain_id, interface):
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except Exception as e:
        return False, f"unitree_sdk2py unavailable: {e}"

    try:
        if interface:
            ChannelFactoryInitialize(domain_id, interface)
        else:
            ChannelFactoryInitialize(domain_id)
        return True, ""
    except Exception as e:
        msg = str(e)
        # Other modules may have already initialized the singleton.
        if "initialized" in msg.lower():
            return True, ""
        return False, msg


def start_odostate_subscriber(topic, shared_state, dds_domain_id, dds_interface):
    try:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import OdoState_
    except Exception as e:
        return False, f"unitree_sdk2py unavailable: {e}"

    ok, err = init_dds_for_odometry(dds_domain_id, dds_interface)
    if not ok:
        return False, f"DDS init failed (domain={dds_domain_id}, interface='{dds_interface}'): {err}"

    def odostate_handler(msg):
        try:
            odom_pos = np.array(msg.position, dtype=np.float64)
            odom_quat_xyzw = np.array(msg.orientation, dtype=np.float64)
            if odom_pos.shape != (3,) or odom_quat_xyzw.shape != (4,):
                return
            quat_norm = np.linalg.norm(odom_quat_xyzw)
            if quat_norm < 1e-8:
                return
            odom_quat_xyzw = odom_quat_xyzw / quat_norm
            odom_quat_wxyz = np.array(
                [odom_quat_xyzw[3], odom_quat_xyzw[0], odom_quat_xyzw[1], odom_quat_xyzw[2]],
                dtype=np.float64,
            )
            with shared_state["lock"]:
                shared_state["odom_pos"] = odom_pos
                shared_state["odom_quat_wxyz"] = odom_quat_wxyz
                shared_state["has_valid_odom"] = True
        except Exception:
            return

    def subscribe_worker():
        try:
            sub = ChannelSubscriber(topic, OdoState_)
            sub.Init(odostate_handler, 1)
            while True:
                time.sleep(0.2)
        except Exception as e:
            with shared_state["lock"]:
                shared_state["subscription_error"] = str(e)

    threading.Thread(target=subscribe_worker, daemon=True).start()
    return True, ""


def terminal_next_listener(stop_event):
    global anim_idx, frame_idx
    print("[TERMINAL] Enter=next motion, p=previous motion, q=quit")
    while not stop_event.is_set():
        try:
            key = input().strip().lower()
        except EOFError:
            break
        if stop_event.is_set():
            break
        if key == "q":
            stop_event.set()
            break
        if key == "p":
            anim_idx -= 1
            frame_idx = 0
            print(f"[TERMINAL] prev -> anim {anim_idx}")
        else:
            anim_idx += 1
            frame_idx = 0
            print(f"[TERMINAL] next -> anim {anim_idx}")

def main(args) -> None:
    global \
        curr_start, \
        num_motions, \
        motion_id, \
        motion_acc, \
        time_step, \
        dt, \
        paused, \
        data_csv_dict, \
        frame_idx, \
        anim_idx, \
        measured_root_state
        
    fps = 50
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, frame_idx, anim_idx = 0, 1, 0, set(), 0, 1 / fps, False, int(0), 0

    def prepend_names(elem, prefix):
        # If element has a 'name' attribute, prepend the prefix
        if 'name' in elem.attrib:
            elem.attrib['name'] = prefix + elem.attrib['name']
        # Recurse for all child elements
        for child in elem:
            prepend_names(child, prefix)

    def replace_attribute(elem, attribute, value):
        # If element has a 'name' attribute, prepend the prefix
        if attribute in elem.attrib:
            elem.attrib[attribute] = value
        # Recurse for all child elements
        for child in elem:
            replace_attribute(child, attribute, value)

    main_scene = etree.parse('g1/scene_empty.xml')
    robot1 = etree.parse('g1/g1_29dof_old.xml')
    robot_asset = robot1.find('asset')
    scene_asset = main_scene.find('asset')
    for mesh in robot_asset.findall('mesh'):
        # INSERT_YOUR_CODE
        mesh.set("file", os.path.join("g1","meshes", mesh.get('file')))
        scene_asset.append(mesh)
    
    robot_default = robot1.find('default')
    scene_default = main_scene.find('default')
    for default in robot_default.findall('default'):
        scene_default.append(default)

    scene_worldbody = main_scene.find('worldbody')
    robot1_body = robot1.find('worldbody').find('body')
    prepend_names(robot1_body, "robot1_")
    scene_worldbody.append(robot1_body)

    robot2 = etree.parse('g1/g1_29dof_old.xml')
    robot2_body = robot2.find('worldbody').find('body')
    prepend_names(robot2_body, "robot2_")
    replace_attribute(robot2_body, "rgba", "0.5 0.1 0.1 1")
    robot2_body.set("pos", "0 -1 -10")
    scene_worldbody.append(robot2_body)

    robot3 = etree.parse('g1/g1_29dof_old.xml')
    robot3_body = robot3.find('worldbody').find('body')
    prepend_names(robot3_body, "robot3_")
    replace_attribute(robot3_body, "rgba", "0.1 0.5 0.1 0.2")
    robot3_body.set("pos", "0 -2 -10")
    scene_worldbody.append(robot3_body)
    
    mj_model = mujoco.MjModel.from_xml_string(etree.tostring(main_scene, pretty_print=True, encoding="unicode"))
    mj_data = mujoco.MjData(mj_model)

    # Disable advanced visual effects for better performance
    mj_model.vis.global_.offwidth = 1920
    mj_model.vis.global_.offheight = 1080
    mj_model.vis.quality.shadowsize = 0  # Disable shadows
    mj_model.vis.quality.offsamples = 1  # Reduce anti-aliasing
    mj_model.vis.rgba.fog = [0, 0, 0, 0]  # Disable fog

    # Disable advanced lighting effects
    mj_model.vis.headlight.ambient = [0.8, 0.8, 0.8]  # Increase ambient light
    mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]  # Increase diffuse light
    mj_model.vis.headlight.specular = [0.1, 0.1, 0.1]  # Reduce specular highlights
    
    terminal_stop_event = threading.Event()

    measured_root_state = None
    effective_measured_root_source = "fixed"

    if args.realtime_debug_url:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(args.realtime_debug_url)
        socket.setsockopt(zmq.SUBSCRIBE, args.realtime_debug_topic.encode())

        data_csv_dicts = [{
            "dof": np.zeros((1,29), dtype=np.float64),
            "root_rot": np.array([[0.0, 0.0, 0.0, 1.0]]),  # [x, y, z, w]
            "root_trans_offset": np.array([[0.0, 0.0, .9]], dtype=np.float64),
            "dof_measured": np.zeros((1,29), dtype=np.float64),
            "root_rot_measured": np.array([[0.0, 0.0, 0.0, 1.0]]),
            "root_trans_offset_measured": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            "vr_3point_position": np.zeros((3,3), dtype=np.float64),
            "vr_3point_orientation": np.zeros((3,4), dtype=np.float64),
            "vr_3point_compliance": np.zeros((3), dtype=np.float64),
        }]

        threading.Thread(
            target=receive_realtime_debug_messages,
            args=(socket, data_csv_dicts, args.realtime_debug_topic),
            daemon=True,
        ).start()

        if args.measured_root_source != "fixed":
            measured_root_state = create_measured_root_state()
            ok, err = start_odostate_subscriber(
                args.odostate_topic,
                measured_root_state,
                args.dds_domain_id,
                args.dds_interface,
            )
            if not ok:
                if args.measured_root_source == "odostate":
                    raise RuntimeError(
                        f"Failed to enable measured root odometry ({args.odostate_topic}): {err}"
                    )
                print(
                    f"[WARN] Failed to subscribe odometry topic '{args.odostate_topic}': {err}. "
                    "Falling back to fixed measured root."
                )
                measured_root_state = None
            else:
                print(
                    "[INFO] OdoState DDS config: "
                    f"domain={args.dds_domain_id}, interface='{args.dds_interface}'"
                )
                effective_measured_root_source = args.measured_root_source
                if args.measured_root_source == "odostate":
                    if not wait_for_odometry(measured_root_state, args.odostate_timeout_sec):
                        raise RuntimeError(
                            "No odometry received on "
                            f"'{args.odostate_topic}' within {args.odostate_timeout_sec:.2f}s."
                        )
                    print(
                        "[INFO] OdoState connected. Measured root will follow odometry "
                        "(with target-aligned reset/start)."
                    )
                elif args.measured_root_source == "auto":
                    if wait_for_odometry(measured_root_state, args.odostate_timeout_sec):
                        print(
                            "[INFO] OdoState connected. Measured root will follow odometry "
                            "(with target-aligned reset/start)."
                        )
                    else:
                        print(
                            "[WARN] No OdoState received within timeout; using fixed measured root "
                            "until odometry becomes available."
                        )

    elif args.motion_dir:
        data_csv_dicts = load_anim_data(args.motion_dir)
    elif args.csv_path:
        data_csv_dicts = load_anim_data(args.csv_path)
    else:
        raise ValueError("Either --realtime_debug_url, --motion_dir, or --csv_path must be provided")

    RECORDING = False
    mj_model.opt.timestep = dt
    try:
        context = mujoco.GLContext(1920, 1080)
        context.make_current()
        print("✓ GPU acceleration enabled")
    except Exception as e:
        print(f"✗ GPU acceleration not available: {e}")
        context = None

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_call_back,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        if args.terminal_next:
            threading.Thread(
                target=terminal_next_listener,
                args=(terminal_stop_event,),
                daemon=True,
            ).start()

        # Set camera position to be further away
        viewer.cam.distance = 15.0  # Increase distance from the scene
        viewer.cam.azimuth = 90.0  # Set azimuth angle
        viewer.cam.elevation = -20.0  # Set elevation angle
        
        while viewer.is_running():
            if terminal_stop_event.is_set():
                break
            motion_len = data_csv_dicts[anim_idx % len(data_csv_dicts)]["dof"].shape[0]
            step_start = time.time()
            time_idx = frame_idx % motion_len
            data_dict = data_csv_dicts[anim_idx % len(data_csv_dicts)]
            mj_data.qpos[:3] = data_dict["root_trans_offset"][time_idx]
            mj_data.qpos[3:7] = data_dict["root_rot"][time_idx]
            mj_data.qpos[7:7+29] = data_dict["dof"][time_idx]

            if "dof_measured" in data_dict:
                display_root_trans_measured = np.array(
                    data_dict["root_trans_offset_measured"][time_idx], dtype=np.float64
                )
                display_root_rot_measured = np.array(
                    data_dict["root_rot_measured"][time_idx], dtype=np.float64
                )

                if (
                    measured_root_state is not None
                    and effective_measured_root_source in ("auto", "odostate")
                ):
                    with measured_root_state["lock"]:
                        has_odom = measured_root_state["has_valid_odom"]
                        odom_pos = measured_root_state["odom_pos"].copy()
                        odom_quat_wxyz = measured_root_state["odom_quat_wxyz"].copy()
                        need_realign = measured_root_state["need_realign"]
                        alignment_offset = measured_root_state["alignment_offset"].copy()
                        reported_odom_active = measured_root_state["reported_odom_active"]
                        reported_realign = measured_root_state["reported_realign"]
                        subscription_error = measured_root_state["subscription_error"]
                        reported_subscription_error = measured_root_state["reported_subscription_error"]

                    if has_odom:
                        if not reported_odom_active:
                            print(
                                "[INFO] OdoState active. Measured root now follows odometry "
                                "(with target-aligned offset)."
                            )
                            with measured_root_state["lock"]:
                                measured_root_state["reported_odom_active"] = True

                        if need_realign:
                            alignment_offset = (
                                np.array(data_dict["root_trans_offset"][time_idx], dtype=np.float64)
                                - odom_pos
                            )
                            with measured_root_state["lock"]:
                                measured_root_state["alignment_offset"] = alignment_offset
                                measured_root_state["need_realign"] = False
                                measured_root_state["reported_realign"] = True
                        elif not reported_realign:
                            with measured_root_state["lock"]:
                                measured_root_state["reported_realign"] = True

                        display_root_trans_measured = odom_pos + alignment_offset
                        display_root_rot_measured = odom_quat_wxyz
                    elif subscription_error and not reported_subscription_error:
                        print(
                            "[WARN] OdoState subscriber error: "
                            f"{subscription_error}. Falling back to fixed measured root."
                        )
                        with measured_root_state["lock"]:
                            measured_root_state["reported_subscription_error"] = True

                mj_data.qpos[36:36+3] = display_root_trans_measured
                mj_data.qpos[39:39+4] = display_root_rot_measured
                mj_data.qpos[43:43+29] = data_dict["dof_measured"][time_idx]


                mj_data.qpos[43+29:43+29+3] = display_root_trans_measured
                mj_data.qpos[43+29+3:43+29+3+4] = data_dict["root_rot"][time_idx]
                mj_data.qpos[43+29+3+4:43+29+3+4+29] = data_dict["dof"][time_idx]

            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                frame_idx += 1
            
            viewer.user_scn.ngeom = 0
            if "vr_3point_position" in data_dict:
                # Get root pose for transforming root-relative coordinates to world space
                # VR 3-point data from C++ is normalized relative to root (see g1_deploy_onnx_ref.cpp)
                if "dof_measured" in data_dict:
                    root_trans = display_root_trans_measured
                    root_quat_wxyz = display_root_rot_measured
                else:
                    root_trans = data_dict["root_trans_offset"][time_idx]
                    root_quat_wxyz = data_dict["root_rot"][time_idx]
                root_rot = R.from_quat(root_quat_wxyz, scalar_first=True)
                
                for i in range(3):
                    # VR 3-point position is in root-relative coordinates, transform to world
                    vr_pos_root_frame = data_dict["vr_3point_position"][i]
                    # vr_pos_world = root_trans + root_rot.apply(vr_pos_root_frame)
                    vr_pos_world = vr_pos_root_frame + root_trans
                    
                    if np.linalg.norm(data_dict["vr_3point_orientation"][i]) > 0:
                        # VR orientation is also root-relative, transform to world
                        # C++ quaternion is in [w, x, y, z] format (scalar_first=True)
                        vr_quat_root_frame = R.from_quat(data_dict["vr_3point_orientation"][i], scalar_first=True)
                        vr_rot_world = root_rot * vr_quat_root_frame  # Quaternion multiplication
                        mat = vr_rot_world.as_matrix()
                    else:
                        mat = root_rot.as_matrix()  # If no VR orientation, use root orientation

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.05, 0.01, 0.01],
                        pos=vr_pos_world,
                        mat=mat.flatten(),
                        rgba=0.5*np.array([1, 1, 0, 2])
                    )
                    viewer.user_scn.ngeom += 1

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    terminal_stop_event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize retargeted motion data in MuJoCo"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Path to the CSV file containing retargeted motion data",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        default="",
        help="Path to the CSV file containing retargeted motion data",
    )
    parser.add_argument(
        "--realtime_debug_url",
        type=str,
        default="",
        help="URL to receive realtime debug messages from",
    )
    parser.add_argument(
        "--realtime_debug_topic",
        type=str,
        default="g1_debug",
        help="Topic to receive realtime debug messages from",
    )
    parser.add_argument(
        "--measured-root-source",
        type=str,
        choices=["auto", "odostate", "fixed"],
        default="auto",
        help=(
            "Measured root source in realtime mode: "
            "auto=use odostate when available, odostate=required, fixed=legacy fixed root."
        ),
    )
    parser.add_argument(
        "--odostate-topic",
        type=str,
        default="rt/odostate",
        help="DDS topic for odometry-based measured root pose.",
    )
    parser.add_argument(
        "--odostate-timeout-sec",
        type=float,
        default=0.5,
        help="Initial wait timeout for odostate (seconds).",
    )
    parser.add_argument(
        "--dds-domain-id",
        type=int,
        default=0,
        help="DDS domain id for odostate subscription (sim default: 0).",
    )
    parser.add_argument(
        "--dds-interface",
        type=str,
        default="lo",
        help="DDS network interface for odostate subscription (sim default: lo).",
    )
    parser.add_argument(
        "--terminal_next",
        action="store_true",
        help="Enable terminal control: Enter=next motion, p=previous, q=quit.",
    )
    args = parser.parse_args()

    main(args)
