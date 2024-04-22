# mf-nerf

Install  
`
python -m pip install -e .
`

`
ns-train mfnerf --data ./scenarios/bww_entrance --vis viewer+wandb --optimizers.fields.scheduler.max-steps 30000 --steps-per-eval-all-images 1000
`

`
ns-train mfnerf --data ./scenarios/bww_entrance/ --vis wandb --experiment-name 9block_bww --steps-per-eval-all-images 1000 --pipeline.model.blocks-x 3 --pipeline.model.blocks-y 3
`


