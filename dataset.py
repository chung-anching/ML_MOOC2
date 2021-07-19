from cv2 import cv2
import numpy as np

def load_data():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(64,64))
        x.append(img)

        if health == "1":
            label = 0
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 2
        elif scab =="1":
            label = 3
        
        y.append(label)

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x.npy",x)
    np.save("train_y.npy",y)
    
    return x,y

def load_data_augmentation():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(64,64))
        
        x.append(img)        
        if health == "1":
            label = 0
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 2
        elif scab =="1":
            label = 3            
        y.append(label)
        
        from numpy import expand_dims
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # brightness augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)
        
        # horizontal shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)        
        
        # vertical shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)   
        
        # rotation augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=80)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  

        # zoom augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label) 

        # extra augmentation
        if multiple_diseases == "1":
            img_hr = cv2.flip(img,1,dst=None) #水平鏡像
            x.append(img_hr)
            y.append(1)
        
            img_vr = cv2.flip(img,0,dst=None) #垂直鏡像
            x.append(img_vr)
            y.append(1)
            
            img_sr = cv2.flip(img,-1,dst=None) #對角鏡像
            x.append(img_sr)
            y.append(1)
            
            # rotation augmentation
            data = img_to_array(img)
            # expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(rotation_range=150)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            for i in range(20):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
                x.append(image)
                y.append(1)  

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_aug.npy",x)
    np.save("train_y_aug.npy",y)
    
    return x,y

img_size = 64

def load_data_rust():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
        
        x.append(img)
        if health == "1":
            label = 0
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 1
        elif scab =="1":
            label = 0
        y.append(label)

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_rust.npy",x)
    np.save("train_y_rust.npy",y)
    
    return x,y

def load_data_rust_unhealthy():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
        
        if health == "1":
            continue
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 1
        elif scab =="1":
            label = 0
        x.append(img)
        y.append(label)

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_rust_unhealthy.npy",x)
    np.save("train_y_rust_unhealthy.npy",y)
    
    return x,y

def load_data_scab():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
        
        x.append(img)
        if health == "1":
            label = 0
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 0
        elif scab =="1":
            label = 1
        y.append(label)

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_scab.npy",x)
    np.save("train_y_scab.npy",y)
    
    return x,y

def load_data_scab_unhealthy():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
        

        if health == "1":
            continue
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 0
        elif scab =="1":
            label = 1
        x.append(img)
        y.append(label)

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_scab_unhealthy.npy",x)
    np.save("train_y_scab_unhealthy.npy",y)
    
    return x,y

def load_data_disease():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
        
        x.append(img)
        if health == "1":
            label = 0
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 1
        elif scab =="1":
            label = 1
        y.append(label)

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_disease.npy",x)
    np.save("train_y_disease.npy",y)
    
    return x,y

times = 3

def load_data_rust_aug():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
        
       
        if health == "1":
            continue
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 1
        elif scab =="1":
            label = 0
            
        x.append(img)             
        y.append(label)
        
        from numpy import expand_dims
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # brightness augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)
        
        # horizontal shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)        
         
        # vertical shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  
         
        # rotation augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=90)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  

        # zoom augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label) 

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_rust_aug.npy",x)
    np.save("train_y_rust_aug.npy",y)
    
    return x,y

def load_data_scab_aug():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
           
        if health == "1":
            continue
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 0
        elif scab =="1":
            label = 1
            
        x.append(img)                
        y.append(label)
        
        from numpy import expand_dims
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # brightness augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)
        
        # horizontal shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)        
         
        # vertical shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)   
         
        # rotation augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=90)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  

        # zoom augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_scab_aug.npy",x)
    np.save("train_y_scab_aug.npy",y)
    
    return x,y

def load_data_disease_aug():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        health = row.split(",")[1]
        multiple_diseases = row.split(",")[2]
        rust = row.split(",")[3]
        scab = row.split(",")[4]
        
        img = cv2.imread("./images/" + filename + ".jpg", cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size,img_size))
        
        x.append(img)        
        if health == "1":
            label = 0
        elif multiple_diseases == "1":
            label = 1
        elif rust == "1":
            label = 1
        elif scab =="1":
            label = 1            
        y.append(label)
        
        from numpy import expand_dims
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # brightness augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)
        
        # horizontal shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)        
         
        # vertical shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)   
         
        # rotation augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=90)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  

        # zoom augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_disease_aug.npy",x)
    np.save("train_y_disease_aug.npy",y)
    
    return x,y