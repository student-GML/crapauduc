This is the main repo of the project crapauduc.
### Structure
The structure is the following:
  - `subset/` contains all the images having a bounding box, the labels are located in the `data/path_and_bounding_box.csv` file

An example of usage is available in the notebook example_use_bounding_box.ipynb

  - `data/` contains the csv describing the images. 
  Be careful as of now the name is wrong; you should replace : by _ in the name of the images!

 The labels are the following : 
 When using yolo we will use the number instead of the value, but as of now the csv has a name and not a number for the class
1. triton
2. grenouille-crapaud
3. planche
4. feuille
5. souris
6. insecte
