import keras
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Activation,Flatten,BatchNormalization,Dropout
from keras.models import Sequential,Model
from keras.optimizers import Adam,SGD,RMSprop
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.layers import GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2

def crate_model(inp_shape,output_sh=15):
    base_model = DenseNet201(include_top=False,weights="imagenet",input_shape=inp_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x=Dense(1024,activation="relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(output_sh,activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model
