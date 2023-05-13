#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

hit = loadFolderImages(hit_path2)
#hit = positive_images

miss = loadFolderImages(miss_path2)
#miss = negative_images


#---------------------------------- Extraccion de caracteristicas -----------------------------------------------------

first_part = []
cont = 1
array = []

for image in hit
    println("Cont:")
    println(cont)
    if(mod(cont,4) == 1)
        global array = []
    end
    element = featureExtraction(image, 0, [0, 1])
    global array = append!(array, element[1:2])

    if(mod(cont, 4) == 0)
        push!(first_part, array)
    end
    global cont = cont + 1
end

cont = 1
array = []
second_part = []

for image in miss
    println("Cont:")
    println(cont)
    if(mod(cont,4) == 1)
        global array = []
    end
    element = featureExtraction(image, 1, [0, 1])
    global array = append!(array, element[1:2])

    if(mod(cont, 4) == 0)
        push!(second_part, array)
    end
    global cont =cont + 1
end

dataSetAux = vcat(first_part, second_part)




#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

hit = loadFolderImages(hit_path)
#hit = positive_images

miss = loadFolderImages(miss_path)
#miss = negative_images

hit1 = loadFolderImages(hit_path1)

miss1 = loadFolderImages(miss_path1)

#---------------------------------- Extracción de características -----------------------------------------------------

first_part = []
second_part = []
third_part = []
fourth_part = []


for image in hit
    temp = featureExtraction(image, 0, [0, 1])
    temp = temp[1:2]
    push!(first_part, temp)
end

for image in miss
    temp = featureExtraction(image, 1, [0, 1])
    temp = temp[1:2]
    push!(second_part, temp)
end

for image in hit1 #Cargo los patrones positivos recortados mas pequeños
    push!(third_part, featureExtraction(image, 0, [0, 1]))
end

for image in miss1#Cargo los patrones negativos recortados mas pequeños
    push!(fourth_part, featureExtraction(image, 1, [0, 1]))
end

dataSet1 = vcat(first_part, second_part)
dataSet2 = vcat(third_part, fourth_part)

dataSet = [] #Ahora mismo es el dataset de la aprox2
i = 1
for fila in dataSet2
    new_fila = append!(dataSet1[i], fila)
    push!(dataSet, new_fila)
    global i = i+1
end



dataSet2 = []
i = 1
for fila in dataSet
    append!(dataSetAux[i], fila)
    #push!(dataSet2, new_fila)
    global i = i+1
end

# Guardamos el dataSet en "aprox3.data"

saveAsData(path*"/"*"aprox3.data", dataSetAux, '\t')

dataSetImportado = readdlm(path*"/"*"aprox3.data", '\t')

print(dataSetAux == dataSetImportado)

salida = open("resultados_aprox_3.txt", "w")

numFolds = 10 

topology=[3,4]
learningRate = 0.01
maxEpochs = 100 #1000
validationRatio = 0.2   # Si queremos que el modelo no se entrene usando un conjunto de validación, este valor debe ser 0
maxEpochsVal = 6
numRepetitions = 5 #50

kernel = "rbf"
kernelDegree = 3
kernelGamma = 2
C=1

maxDepth = 4

numNeighbors = 8

#Dividimos el dataset en dos (features y targets)
# Longitud de cada fila de "dataSetImportado"
numCaracteristicas = size(dataSetImportado, 2)
inputs = convert(Array{Float64, 2}, dataSetImportado[:, 1:2]) 
targets = reshape((dataSetImportado[:, numCaracteristicas]), :, 1)

#PDF's
# inputs = dataSetImportado[:, 1:2];
# inputs = convert(Array{Float32, 2}, inputs);
# targets = dataSetImportado[:, 3];
# targets = convert(Array{Bool, 1}, targets);

normalizeMinMax!(inputs);

# Dicionario con los hiperparametros utilizados para el entrenamiento de RNA

modelHyperparameters = Dict();
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitions;
modelHyperparameters["maxEpochs"] = maxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;


# RNA

topologies = [[3], [1], [2], [1,1], [1,2], [3,4], [2,3], [5,5]]

for i in topologies
    modelHyperparameters["topology"] = i
    macc, sacc, mrecall, srecall = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, numFolds);
    write(salida, string("\nRNA: ", string(i), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), "\nMedia de Sensibilidad de test ", round(mrecall*100, digits=2), "\nDesviacion tipica de Sensibilidad de test ", round(srecall*100, digits=2)))
end

# Hiperparametros utilizados para el entrenamiento de SVM

modelHyperparameters = Dict();
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;


# SVM

topologies = [("rbf", 0.1), ("rbf", 0.6), ("rbf", 1), ("poly", 0.1), ("poly", 0.6), ("poly", 1), ("sigmoid", 0.1), ("sigmoid", 0.6), ("sigmoid", 1)]

for topology in topologies
    modelHyperparameters["kernel"] = topology[1];
    modelHyperparameters["C"] = topology[2];
    macc, sacc, mrecall, srecall = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
    write(salida, string("\nSVM: con kernel \"", string(topology[1]), "\" y C:", string(topology[2]), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), "\nMedia de Sensibilidad de test ", round(mrecall*100, digits=2), "\nDesviacion tipica de Sensibilidad de test ", round(srecall*100, digits=2)))
end


# Arbol de Decision

depths = [1, 3, 4, 5, 6, 7, 10]

for depth in depths
    macc, sacc, mrecall, srecall = modelCrossValidation(:DecisionTree, Dict("maxDepth" => depth), inputs, targets, numFolds);
    write(salida, string("\nDECISION TREE: ", string(depth), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), "\nMedia de Sensibilidad de test ", round(mrecall*100, digits=2), "\nDesviacion tipica de Sensibilidad de test ", round(srecall*100, digits=2)))
end


# KNN

neighboors = [2,5, 8, 11, 14, 17]

for i in neighboors
    macc, sacc, mrecall, srecall= modelCrossValidation(:kNN, Dict("numNeighbors" => i), inputs, targets, numFolds);
     write(salida, string("\nKNN: ", string(i), "\nMedia de precision del test en 10-fold: ",round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), "\nMedia de Sensibilidad de test ", round(mrecall*100, digits=2), "\nDesviacion tipica de Sensibilidad de test ", round(srecall*100, digits=2)))
end


close(salida)
