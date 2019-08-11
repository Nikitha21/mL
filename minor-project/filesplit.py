train = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/train_data.csv")
train1 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data1.csv", 'w')
train2 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data2.csv", 'w')
train3 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data3.csv", 'w')
train4 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data4.csv", 'w')
train5 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data5.csv", 'w')
train6 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data6.csv", 'w')
train7 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data7.csv", 'w')
train8 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data8.csv", 'w')
train9 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data9.csv", 'w')
train10 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data10.csv", 'w')
train11 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data11.csv", 'w')
train12 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data12.csv", 'w')
train13 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data13.csv", 'w')
train14 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data14.csv", 'w')
train15 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data15.csv", 'w')
train16 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data16.csv", 'w')
train17 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data17.csv", 'w')
train18 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data18.csv", 'w')
train19 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data19.csv", 'w')
train20 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data20.csv", 'w')
train21 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data21.csv", 'w')
train22 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data22.csv", 'w')
train23 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data23.csv", 'w')
train24 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data24.csv", 'w')
train25 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data25.csv", 'w')
train26 = open("F:/New folder/NIKITHA RAO/Desktop/7th sem/mini-project/Parkinson_Multiple_Sound_Recording_Data/data_set/train_data26.csv", 'w')

'''test1 = open("test_data1.csv", 'w')
test2 = open("test_data2.csv", 'w')
test3 = open("test_data3.csv", 'w')
test4 = open("test_data4.csv", 'w')
test5 = open("test_data5.csv", 'w')
test6 = open("test_data6.csv", 'w')
test7 = open("test_data7.csv", 'w')
test8 = open("test_data8.csv", 'w')
test9 = open("test_data9.csv", 'w')
test10 = open("test_data10.csv", 'w')
test11 = open("test_data11.csv", 'w')
test12 = open("test_data12.csv", 'w')
test13 = open("test_data13.csv", 'w')
test14 = open("test_data14.csv", 'w')
test15 = open("test_data15.csv", 'w')
test16 = open("test_data16.csv", 'w')
test17 = open("test_data17.csv", 'w')
test18 = open("test_data18.csv", 'w')
test19 = open("test_data19.csv", 'w')
test20 = open("test_data20.csv", 'w')
test21 = open("test_data21.csv", 'w')
test22 = open("test_data22.csv", 'w')
test23 = open("test_data23.csv", 'w')
test24 = open("test_data24.csv", 'w')
test25 = open("test_data25.csv", 'w')
test26 = open("test_data26.csv", 'w')'''

line_num = 0

for line in train:
    if line_num < 1040:
        if line_num % 26 == 0:
            train1.write(line)
        elif line_num % 26 == 1:
            train2.write(line)
        elif line_num % 26 == 2:
            train3.write(line)
        elif line_num % 26 == 3:
            train4.write(line)
        elif line_num % 26 == 4:
            train5.write(line)
        elif line_num % 26 == 5:
            train6.write(line)
        elif line_num % 26 == 6:
            train7.write(line)
        elif line_num % 26 == 7:
            train8.write(line)
        elif line_num % 26 == 8:
            train9.write(line)
        elif line_num % 26 == 9:
            train10.write(line)
        elif line_num % 26 == 10:
            train11.write(line)
        elif line_num % 26 == 11:
            train12.write(line)
        elif line_num % 26 == 12:
            train13.write(line)
        elif line_num % 26 == 13:
            train14.write(line)
        elif line_num % 26 == 14:
            train15.write(line)
        elif line_num % 26 == 15:
            train16.write(line)
        elif line_num % 26 == 16:
            train17.write(line)
        elif line_num % 26 == 17:
            train18.write(line)
        elif line_num % 26 == 18:
            train19.write(line)
        elif line_num % 26 == 19:
            train20.write(line)
        elif line_num % 26 == 20:
            train21.write(line)
        elif line_num % 26 == 21:
            train22.write(line)
        elif line_num % 26 == 22:
            train23.write(line)
        elif line_num % 26 == 23:
            train24.write(line)
        elif line_num % 26 == 24:
            train25.write(line)
        elif line_num % 26 == 25:
            train26.write(line)
    line_num = line_num + 1

train1.close()
train2.close()
train3.close()
train4.close()
train5.close()
train6.close()
train7.close()
train8.close()
train9.close()
train10.close()
train11.close()
train12.close()
train13.close()
train14.close()
train15.close()
train16.close()
train17.close()
train18.close()
train19.close()
train20.close()
train21.close()
train22.close()
train23.close()
train24.close()
train25.close()
train26.close()
