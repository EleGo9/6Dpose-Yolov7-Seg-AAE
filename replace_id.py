import csv
import yaml
import pandas as pd

#file = '/home/elena/repos/6Dpose-Yolov7-Seg-AAE/aaeyolov4-4objplane.csv'
file = '/home/elena/repos/bop_toolkit/cifarelli_eval/yolov4_chiavecandela-plane.csv'
#file = '/home/elena/repos/Cifarelli/6d_test_plane/gt.yml'
csvfile = True
ymlfile = False

if csvfile:
        df = pd.read_csv(file)

        # Elimino obj id 2 e obj id 4
        df =  df[df.scene_id != 2] 
        df =  df[df.scene_id != 4] 

        # Cambio gli obj id sbagliati
        df = df.replace({'scene_id': 3, 'obj_id': 3}, 2)
        df = df.replace({'scene_id': 5, 'obj_id': 5}, 3)
        df = df.replace({'scene_id': 6, 'obj_id': 6}, 4)
        df = df.replace({'scene_id': 1, 'obj_id': 1}, 0)
        print(df['scene_id'])

        df.to_csv('/home/elena/repos/bop_toolkit/cifarelli_eval/yolov4_chiavecandela-plane2.csv')

elif ymlfile:
        # opening a file
        with open(file, 'r') as stream:
                try:
                # Converts yaml document to python object
                        d=yaml.safe_load(stream)
                        # print('d keys',d.keys())
                        # print('d of 0 of 0 obj_id',d[0][0]['obj_id'])
                        df = pd.DataFrame.from_dict(d)
                        print('pandas df created')
                except yaml.YAMLError as e:
                        print(e)

        

        for image in d.values():
                        print('image', image)
                        for row in image:
                                if row['obj_id']==1:
                                        row['obj_id']=2
                                elif row['obj_id']==2:
                                        row['obj_id']=3
                                elif row['obj_id']==3:
                                        row['obj_id'] = 4
                                elif row['obj_id'] == 6:
                                        row['obj_id']= 0
                        


        with open('/home/elena/repos/Cifarelli/6d_test_plane/gt2.yml','w') as f:
                yaml.dump(d, f)


                        


























