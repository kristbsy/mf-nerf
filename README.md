# mf-nerf

Install  
`
pip install -e .
`

# Image segmentation and masking
Remove vehicles and people from images and replace with transparency  

Install requirements:  
`pip install -r ./seg/requirements.txt`

Inplace process all images (only png) in folder specified.  
`python -m seg.process [datafolder]`