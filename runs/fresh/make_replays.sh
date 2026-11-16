#!/bin/zsh
# Turn every snapshot into a replay page as soon as it appears; stop when training ends.
cd "$(dirname "$0")/../.."
while true; do
  for ck in runs/fresh/iter_*.pt(N); do
    html=${ck:r}.html; html=${html/iter_/replay_iter_}
    if [[ ! -f $html ]]; then
      sleep 2   # let the save finish
      .venv/bin/python replay.py --ckpt $ck --birds 4 --seconds 60 --out $html 2>&1 | grep -v -i warn | sed "s|^|[${ck:t:r}] |"
    fi
  done
  if ! pgrep -f "train.py --out runs/fresh" >/dev/null; then
    for ck in runs/fresh/iter_*.pt(N); do h=${ck:r}.html; h=${h/iter_/replay_iter_}; [[ -f $h ]] || continue 2; done
    grep -E "Traceback|Error" runs/fresh/train.log | head -3; echo "training finished"; break
  fi
  sleep 10
done
