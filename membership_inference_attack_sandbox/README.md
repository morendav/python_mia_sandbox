# Membership Inference Attack Sandbox

Datasets
* uses some available through tf datasets (e.g. cifar10)
* other datasets will be attributed to source and licence in the dataset readme file (if this is out of sync feel free to nudge me to add licences here)


Codebase 

1. mia_intro_image_class.py
   * MIA using a custom dataset (from dataset dir)
   * runs MIA as a callback during model training
   * two models are built and tested at some parameterized frequency (each_epoch)

TECH DEBT: this doesn't produce expected results. As number of training epochs increases the models do become over fit, 
I would expect this to contribute to a growing attacker advantage in MIA results. Instead the results are erratic, 
and highly variable.


2. mia_tutorial.py
   * this uses similar methods to #1, but uses an out of the box dataset (cifar)
   * tracks very closely with the MIA tutorial on tensorflow privacy, possibly minor edits in place

3. mod-mia_tutorial.py
   * this is mia tutorial but uses the kaggle data
   * RESULTS: were erratic also

4. mode-mia_intro_image_class-cifar10.py
   * This is #1, however instead of using kaggle data I've swapped in place edits for the tutorial script that use the cifar 10 datasets
   * I suspected that the batching, or the format of the import datasets in #1 was to blame for the mia results from #1

5. [mia_intro_imageclassifier.py](mia_intro_imageclassifier.py)
   * this is #1 but going to use either cifar or another off shelf dataset
   * includes builder methods for model compilation