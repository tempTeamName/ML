def train(x_train, y_train, path):

    # logisticReg
    C = [1, 10, 100]
    for c in C:   
        start_time = time.time()
        model = logisticReg(x_train, y_train,c)
        printStr = f'logisticReg c = {c} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr, path)
        pickle.dump(model, open(path+f"logisticReg{c}.sav", 'wb'))


    # SVM linear
    start_time = time.time()
    model = linearSvm(x_train, y_train)
    printStr = 'linear svm time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+"linearSvm.sav", 'wb'))

    # poly SVM
    degree = [2, 3]
    for deg in degree:
        start_time = time.time()
        model = polySvm(x_train, y_train, deg)
        printStr = f'polySvm degree = {deg} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr, path)
        pickle.dump(model, open(path+f"polySvm{deg}.sav", 'wb'))

    # gaussian SVM
    start_time = time.time()
    model = gaussianSvm(x_train, y_train)
    printStr = "gaussianSvm time : "+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+"gaussianSvm.sav", 'wb'))

    # decision tree
    level = [3, 5, 7]
    for lev in level:
        start_time = time.time()
        model = dTree(x_train, y_train, lev)
        printStr = f'Decision Tree level = {lev} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr, path)
        pickle.dump(model, open(path+f"dTree{lev}.sav", 'wb'))

