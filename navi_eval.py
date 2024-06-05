def navi_eval(navi_path):
    navi_cat = 10
    ''' 0   1   2
        3   4   5
        6   7   8
        以上分别代表不同的方向，其中4为停止
        9为此次未识别，保持上一次的值
        10为初始值


    if x_delta > 0:
        if y_delta > 0:
            navi_cat = 2
        elif y_delta == 0:
            navi_cat = 5
        else:
            navi_cat = 8
    elif x_delta == 0:
        if y_delta > 0:
            navi_cat = 1
        elif y_delta == 0:
            navi_cat = 4
        else:
            navi_cat = 7
    else:
        if y_delta > 0:
            navi_cat = 0
        elif y_delta == 0:
            navi_cat = 3
        else:
            navi_cat = 6

    return navi_cat

if __name__ == "__main__":
    navi_path = [(5,5), (7,3)]
    print(navi_eval(navi_path))
