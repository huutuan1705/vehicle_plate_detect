char_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',\
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

vehicle = ['car','threewheel','bus','truck','motorbike','van']

def predict_vehicle(model, image):
    res = []
    results = model(image)
    if len(results.xyxy[0]) > 0 :
        for i in range(len(results.xyxy[0])) :
            left, top, right, bottom, score, label =int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3]), float(results.xyxy[0][i][4]), int(results.xyxy[0][i][5])
            if score >= 0.45:
                res.append((left, top, right, bottom, score, vehicle[label]))
    return res

def predict_plate(model, image):
    res = []
    results = model(image)
    # print(results.xyxy[0])
    if len(results.xyxy[0]) > 0 :
        for i in range(len(results.xyxy[0])) :
            left, top, right, bottom, score, label =int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3]), float(results.xyxy[0][i][4]), int(results.xyxy[0][i][5])
            if score >= 0.45:
                width = right - left
                height = bottom - top
                res.append(([left, top, width, height], score, 'plate'))
    return res

def predict_character(model, image):
    res = []
    height, width = image.shape[:2] 
    if width > 2*height:
        results = model(image)
        if len(results.xyxy[0]) > 0 :
            for i in range(len(results.xyxy[0])) :
                left, top, right, bottom, score, label =int(results.xyxy[0][i][0]),\
                    int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3]),\
                            float(results.xyxy[0][i][4]), int(results.xyxy[0][i][5])
                res.append((left, top, right, bottom, score, label))
            res.sort(key=lambda x: x[0])
    else:
        new_height = height // 2

        upper_half = image[:new_height, :]
        lower_half = image[new_height:, :]

        results1 = model(upper_half)
        results2 = model(lower_half)

        if len(results1.xyxy[0]) > 0 :
            res1 = []
            for i in range(len(results1.xyxy[0])) :
                left, top, right, bottom, score, label =int(results1.xyxy[0][i][0]),\
                    int(results1.xyxy[0][i][1]),int(results1.xyxy[0][i][2]),int(results1.xyxy[0][i][3]),\
                            float(results1.xyxy[0][i][4]), int(results1.xyxy[0][i][5])
                res1.append((left, top, right, bottom, score, label))
            res1.sort(key=lambda x: x[0]) 
            res += res1

        if len(results2.xyxy[0]) > 0 :
            res2 = []
            for i in range(len(results2.xyxy[0])) :
                left, top, right, bottom, score, label =int(results2.xyxy[0][i][0]),\
                    int(results2.xyxy[0][i][1]),int(results2.xyxy[0][i][2]),int(results2.xyxy[0][i][3]),\
                            float(results2.xyxy[0][i][4]), int(results2.xyxy[0][i][5])
                res2.append((left, top, right, bottom, score, label))
            res2.sort(key=lambda x: x[0])  
            res += res2
    
    plate = ""
    for i in res:
        plate += char_label[i[5]]
    return plate