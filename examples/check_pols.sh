
echo "Checking run 0 with run 0"
python ./examples/validate_pol.py ./prov/torchless_run_0 ./prov/torchless_run_0 relative
echo "Checking run 0 with run 1 (different seed)"
python ./examples/validate_pol.py ./prov/torchless_run_0 ./prov/torchless_run_1 relative
echo "Checking run 0 with run 2 (different lr)"
python ./examples/validate_pol.py ./prov/torchless_run_0 ./prov/torchless_run_2 relative
echo "Checking run 0 with run 3 (different loss)"
python ./examples/validate_pol.py ./prov/torchless_run_0 ./prov/torchless_run_3 relative