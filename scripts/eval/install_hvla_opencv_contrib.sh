#!/usr/bin/env bash
# Install opencv-contrib-python at the pin in maniflow-tb/requirements.txt.
# EJBL depth augmentation (evaluate.py --depth_aug ejbl) needs cv2.ximgproc,
# which is not in the stock opencv-python wheel. Training uses the same pin via
# HVLA requirements and the maniflow-opencv-contrib Beaker image; eval jobs on
# toyblocks-main-maniflow install it at runtime when DEPTH_AUG is set.

install_hvla_opencv_contrib() {
  local req_file="${H_VLA_PATH}/requirements.txt"
  if [ ! -f "$req_file" ]; then
    echo "install_hvla_opencv_contrib: missing ${req_file}" >&2
    return 1
  fi
  local pin
  pin="$(grep -E '^opencv-contrib-python==' "$req_file" | head -n1 || true)"
  if [ -z "$pin" ]; then
    echo "install_hvla_opencv_contrib: no opencv-contrib-python== pin in ${req_file}" >&2
    return 1
  fi
  echo "===> installing ${pin} from HVLA requirements (DEPTH_AUG=${DEPTH_AUG:-})"
  # Only one OpenCV wheel family can be installed at a time.
  uv pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
  uv pip install "$pin"
}
