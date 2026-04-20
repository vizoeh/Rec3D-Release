# Rec3D
3D reconstruction from Z stack (for regular and irregular spacing)\
It uses three main libraries: CSV, PIL and PyVista.\
The reconstruction is based on a cube composed of 6 different images made from the stack.
## Images and *Z-Stacks*
The first important thing to note when using this code is that **every** image should have its name composed of two numbers and be formatted to png ("[x][x].png", ex.: *00.png; 01.png; 02.png [...]*).\
You can put the images on */image_stack/* directly, or put them in */image_stack_pre/* then run *image_equalizer.py*, which will average brightness levels for each image based on one image of the stack and save the outputs to */image_stack/* so you can run *gen_volume.py*
## Depth Data
In "**depth.csv**", you can find a csv file containing two columns: name and depth. The column "*name*" is functionally useless, it's purpose is to help organize and check if all different depths are correct. The column "*depth*" must be filled with float or integers numbers, that will later be converted to integers by multiplying each depth by a factor (you can change the factor *fac* in **csv_reader.py**).\
The depth function **relative** to the top-most number, associated with image 00, which means that each subsequent number must be **lesser than the previous**.
