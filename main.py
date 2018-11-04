from model import *
from data import *
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,str(sys.argv[1]),'image','label',data_gen_args,save_to_dir = 'aug')

model = unet()
model_checkpoint = ModelCheckpoint('unet_lung.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator(str(sys.argv[2]), num_image=len(os.listdir(str(sys.argv[2]))))
results = model.predict_generator(testGene,len(os.listdir(str(sys.argv[2]))),verbose=1)
saveResult(str(sys.argv[2]),results)