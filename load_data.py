
def get_features_and_labels_validation(img2_shape,val_size=0.1):
    train_csv=pd.read_csv("train_target.csv")
    df_label=pd.pivot_table(train_csv, values='PredictedOutcome', index=["ImageName"],
                        columns=['CareSymbolTag'])
    df_label.columns.name=None
    df_label=df_label.reset_index()
    df_label["idx"]=df_label["ImageName"].apply(lambda x: x.split("_")[1])
    df_label.index=df_label.idx
    del df_label["idx"]



    train_list=sorted(["train_cropped/"+i for i in os.listdir("train_cropped")])
    df_feature=[]
    for pic in train_list:
        name=pic.split(".")[0].split("/")[1]
        im=cv2.imread(pic)
        if int(name.split("_")[1])%50==0:
            print(name)
        im=cv2.resize(im,img2_shape[0:2])
    #         im=histogram_equalize(im)
    #         im = cv2.filter2D(im, -1, kernel_sharpening)
        df_feature.append(im)

    del df_label["ImageName"]
    df_label=df_label.values
    idx_tbt=np.random.choice(list(range(len(df_label))),int(len(df_label)*val_size))
    val_names=np.array([i.split(".")[0].split("/")[1] for i in train_list])[idx_tbt]
    val_X_2=np.array([df_feature[i] for i in idx_tbt])
    val_y_2=df_label[idx_tbt]
    df_feature=[df_feature[i] for i in range(len(df_label)) if not i in idx_tbt]
    df_label=np.array([df_label[i] for i in range(len(df_label)) if not i in idx_tbt])
    synd_list=["synd_cropped/"+i for i in os.listdir("synd_cropped")]
    synd_label=[]
    synd_feat=[]
    for pic in synd_list:
        im=cv2.imread(pic)
        name=pic.split("-")[0].split("/")[-1]
        print(name)
        im=cv2.resize(im,img2_shape[0:2])
        synd_feat.append(im)
        lalbl=list(map(int,name.split("_")))
        synd_label.append(lalbl)
    [df_feature.append(i) for i in synd_feat]
    df_feature=np.array(df_feature)
    df_label=df_label.tolist()
    [df_label.append(i) for i in synd_label]
    df_label=np.array(df_label)
    return df_feature,df_label,val_X_2,val_y_2
