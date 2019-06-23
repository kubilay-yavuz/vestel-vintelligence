import keras.backend as K
from tensorflow import set_random_seed
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import RMSprop,SGD,Adam
from .utils import hn_multilabel_loss,recall_m,precision_m,f1_m,focal_loss
from .load_data import get_features_and_labels_validation
from .model import create_model

K.clear_session()
np.random.seed(1337)
set_random_seed(1337)

labels="bleach_with_non_chlorine, do_not_bleach, do_not_dryclean, do_not_tumble_dry, do_not_wash, double_bar, dryclean, low_temperature_tumble_dry, normal_temperature_tumble_dry, single_bar, tumble_dry, wash_30, wash_40, wash_60, wash_hand"
labels=labels.split(", ")
img2_shape=(int(360),int(360),3)
X_train, y_train, val_X_2, val_y_2=get_features_and_labels_validation(img2_shape,val_size=0.2)
a=list(range(len(y_train)))
np.random.shuffle(a)
y_train=y_train[a]
X_train=X_train[a]

cb=Callback()
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_hn_multilabel_loss', verbose=1, save_best_only=True, mode='min')
reduceLROnPlat = ReduceLROnPlateau(monitor='val_hn_multilabel_loss', factor=0.1,
                                   patience=3, verbose=1, mode='min', cooldown=2, min_lr=0.000000001)
erly_s=EarlyStopping(monitor='val_hn_multilabel_loss',patience=7)
cb_list=[cb,
         checkpoint,
         erly_s,
         reduceLROnPlat]
image_gen = ImageDataGenerator(
   # featurewise_center=True,
#     featurewise_std_normalization=True,
    rotation_range=360,
    shear_range=0.4,
    width_shift_range=.15,
    height_shift_range=.15,
    zoom_range=0.5,
#   preprocessing_function=preprocess_input
    # zca_whitening=True
)
model=create_model(inp_shape=img2_shape,output_sh=15)

image_gen.fit(X_train, augment=True)
model.compile(optimizer=RMSprop(lr=0.0001), loss="binary_crossentropy",
              metrics=['acc',hn_multilabel_loss,precision_m,recall_m,f1_m,focal_loss])
model.load_weights("weight_with_bb.hdf5")
history=model.fit_generator(image_gen.flow(X_train,y_train,batch_size=16), epochs=30, steps_per_epoch=X_train.shape[0]//16,validation_data=(val_X_2, val_y_2),callbacks=cb_list,workers=-1)
