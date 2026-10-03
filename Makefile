PY ?= python3
install:      ; $(PY) -m pip install -r requirements.txt
test:         ; $(PY) -m unittest discover -s tests -v
selftest:     ; $(PY) -m roomscan selftest
sim:          ; $(PY) -c "from roomscan.sim.apartment import simulate_capture; simulate_capture('data/sim_demo', n_frames=500, seed=1)"
demo:         sim ; $(PY) -m roomscan run data/sim_demo -o out/sim_demo --gt data/sim_demo/ground_truth_scored.json || $(PY) -m roomscan run data/sim_demo -o out/sim_demo
compliance:   ; $(PY) scripts/compliance_check.py
