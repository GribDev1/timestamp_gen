"""
generate and submit to the scheduler slurm scripts for ToF dynamics

uses slurm template
the slurm template run a python file run_slurm_script.py
"""
import subprocess
import os

slurm_template = 'run_pixels.slurm' # TODO: Create File

test_num = 0

py = 0

for px in range(8): # ToF Width
    for py in range(8): # ToF Height
            with open(slurm_template, 'r') as file:
                filedata = file.read()
                # Replace target string
                filedata = filedata.replace('pixel_x', str(px))
                filedata = filedata.replace('pixel_y', str(py))

                # Write file out again
                slurm_out = f'run_{test_num}_temp.slurm'
                with open(slurm_out, 'w') as file:
                    file.write(filedata)

                p = subprocess.run(['sbatch', f'{slurm_out}'])
                print(p)
                test_num += 1       