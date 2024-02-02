import cv2


class DrawAnglesOnImage:
    @staticmethod
    def draw(result, objects_center, objects_z_rotation, rotations, chosen_object_index):
        for i, (oc, ozr, r) in enumerate(zip(objects_center, objects_z_rotation, rotations)):
            # 90-math.degrees(all_rotation_deg[0][2])
            # rot_z = math.degrees(rot_z)

            '''result = cv2.line(result, oc, r[0], (100, 0, 0), 3)
            result = cv2.line(result, oc, r[1], (0, 100, 0), 3)
            result = cv2.line(result, oc, r[2], (0, 0, 100), 3)'''
            div = 5
            result = cv2.line(result, oc, tuple([oc[0]+(r[2][0]-oc[0])//div, oc[1]+(r[2][1]-oc[1])//div]), (0, 0, 255), 2)
            result = cv2.line(result, oc, tuple([oc[0]+(r[1][0]-oc[0])//div, oc[1]+(r[1][1]-oc[1])//div]), (0, 255, 0), 2)
            result = cv2.line(result, oc, tuple([oc[0]+(r[0][0]-oc[0])//div, oc[1]+(r[0][1]-oc[1])//div]), (255, 0, 0), 2)

            #result = cv2.line(result, oc, r[2]//div, (0, 0, 255), 2)
            #result = cv2.line(result, oc, r[1]//div, (0, 255, 0), 2)
            #result = cv2.line(result, oc, r[0]//div, (255, 0, 0), 2)
            color = (50, 50, 50)
            if i == chosen_object_index:
                color = (50, 200, 255)

            cv2.circle(result, oc, radius=5, color=color, thickness=-1)
            # text = "z: {:.0f}".format(int(ozr)) # (int(ozr) if int(ozr) >= 0 else (360+int(ozr)))
            # cv2.putText(result, text, [oc[0]+10, oc[1]-10], cv2.FONT_ITALIC, 0.6, (250, 170, 40), 2)

    @staticmethod
    def draw_z(result, objects_center, objects_z_rotation):
        for i, (oc, ozr) in enumerate(zip(objects_center, objects_z_rotation)):
            # 90-math.degrees(all_rotation_deg[0][2])
            # rot_z = math.degrees(rot_z)
            result = cv2.ellipse(result, oc, (50, 1),
                                 ozr, 0, 360,
                                 (0, 0, 100), -1)
            result = cv2.ellipse(result, oc, (30, 1),
                                 ozr+90, 0, 360,
                                 (30, 200, 255), -1)
            cv2.circle(result, oc, radius=5, color=(0, 0, 180), thickness=-1)
            cv2.putText(result, str(int(ozr)), [i + 10 for i in oc], cv2.FONT_ITALIC, 0.7, (30, 30, 30), 2)
