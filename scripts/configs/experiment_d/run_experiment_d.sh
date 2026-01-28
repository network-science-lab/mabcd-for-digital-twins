for i in {1..5}; 
    do echo "Running experiment for d=$i"; 
    uv run mfdt "scripts/configs/experiment_d/experiment_freebase_generate_rudimentary_d${i}.yaml"; 
done
uv run mfdt scripts/configs/experiment_d/experiment_freebase_evaluate_rudimentary_d.yaml