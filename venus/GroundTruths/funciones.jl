include("Practica4.jl")


# PATHS

pattern_path = pwd()*"/venus/GroundTruths/Patrones"
img_path = pwd()*"/venus/imagenes"

# Imágenes recortadas a mano
img_path4 = pwd()*"/venus/cuts"
hit_path = pwd()*"/venus/hit"
miss_path = pwd()*"/venus/miss"
hit_path2 = pwd()*"/venus/hit2"
miss_path2 = pwd()*"/venus/miss2"
hit_path3 = pwd()*"/venus/hit3"
miss_path3 = pwd()*"/venus/miss3"
miss_path_4 = pwd()*"/venus/miss_4"
miss_path2_4 = pwd()*"/venus/miss2_4"
miss_path3_4 = pwd()*"/venus/miss3_4"
cuts_path = pwd()*"/venus/cuts"
path = pwd()*"/venus"


K_MULTIPLIER = 1 #Esto sirve para el radio que se usa para recortar

"""
    imagen: Imagen origen que se va a recortar
    porcentaje: Autoexplicativo
"""
function recortar(imagen, porcentaje::Float64)
    #imshow(imagen)
    
    image_size = size(imagen)

    #Dividimos el tamaño de la imagen a la mitad para dividir los ejes

    middle_x = Int(round((image_size[1])/2))
    middle_y = Int(round((image_size[2])/2))
    
    start_x = Int(round(middle_x-(image_size[1]*porcentaje/2)))
    end_x = Int(round(middle_x+(image_size[1]*porcentaje/2)))
    start_y = Int(round(middle_y-(image_size[2]*porcentaje/2)))
    end_y = Int(round(middle_y+(image_size[2]*porcentaje/2)))

    # Las imágenes están traspuestas, habrá que recortar acorde a ello y trasponer
    # el resultado
    #= #! transpose warning
        This operation is intended for linear algebra usage
        - for general data manipulation see [permutedims](@ref Base.permutedims),
        which is non-recursive
    =#
    imagen2 = imagen[start_x:end_x, start_y:end_y]
    #imagen2 = transpose(imagen2)
    return imagen2
end

"""
    imagen: Imagen origen que se va a recortar
    (x, y): Tamaño del rectángulo que se quiere recortar
"""
function recortar(imagen, (x, y)::Tuple)
    #imshow(imagen)
    
    image_size = size(imagen)
    # El radio se redondea hacia arriba para abarcar más area de la que abarcariamos de poder considerar los flotantes
    #Dividimos el tamaño de la imagen a la mitad para dividir los ejes

    middle_x = Int(round((start_x+end_x)/2))
    middle_y = Int(round((start_y+end_y)/2))
    
    start_x = middle_x-x
    end_x = middle_x+x
    start_y = middle_y-y
    end_y = middle_y+y

    # Las imágenes están traspuestas, habrá que recortar acorde a ello y trasponer
    # el resultado
    #= #! transpose warning
        This operation is intended for linear algebra usage
        - for general data manipulation see [permutedims](@ref Base.permutedims),
        which is non-recursive
    =#
    imagen2 = imagen[start_y:end_y, start_x:end_x]
    #imagen2 = transpose(imagen2)
    return imagen2
end

"""
    coords: [X0, Y0, radius]
    imagen: Imagen origen que se va a recortar
    k: Constante que multiplica al radio, por defecto tiene el valor 1
"""
function recortar(coords, imagen)

    #imshow(imagen)
    
    # El radio se redondea hacia arriba para abarcar más area de la que abarcariamos de poder considerar los flotantes
    image_size = size(imagen)
    rounded_radius = Int(ceil(coords[3])) * K_MULTIPLIER

    start_x = Int(round(max(1, coords[1] - rounded_radius)))
    end_x = Int(round(min(coords[1]+rounded_radius, image_size[1])))
    start_y = Int(round(max(1, coords[2] - rounded_radius)))
    end_y = Int(round(min(coords[2]+rounded_radius, image_size[2])))
    
    println("recortar-> Recortando:\t("*string(start_x)*","*string(start_y)*")-----"*"("*string(end_x)*","*string(end_y)*")")

    # Las imágenes están traspuestas, habrá que recortar acorde a ello y trasponer
    # el resultado
    #= #! transpose warning
        This operation is intended for linear algebra usage
        - for general data manipulation see [permutedims](@ref Base.permutedims),
        which is non-recursive
    =#
    imagen2 = imagen[start_y:end_y, start_x:end_x]
    #imagen2 = transpose(imagen2)
    return imagen2
end


"""
    imagen: Imagen origen que se va a recortar en 4 imagenes
"""
function recortar2(imagen)

    #imshow(imagen)

    recortes = []
    
    image_size = size(imagen)

    #Dividimos el tamaño de la imagen a la mitad para dividir los ejes
    
    start_x = 1
    end_x = image_size[1]
    start_y = 1
    end_y = image_size[2]

    middle_x = Int(round((image_size[1])/2))
    middle_y = Int(round((image_size[2])/2))

    if(end_x == 0 || end_y == 0 || middle_x == 0 || middle_y == 0)
        error("Error... Invalid image")
    end

    println("recortar-> Recortando")

    # Las imágenes están traspuestas, habrá que recortar acorde a ello y trasponer
    # el resultado
    #= #! transpose warning
        This operation is intended for linear algebra usage
        - for general data manipulation see [permutedims](@ref Base.permutedims),
        which is non-recursive
    =#
    recorte1 = imagen[1:middle_x, 1:middle_y]
    recorte2 = imagen[middle_x:end_x, 1:middle_y]
    recorte3 = imagen[1:middle_x, middle_y:end_y]
    recorte4 = imagen[middle_x:end_x, middle_y:end_y]

    push!(recortes,recorte1);
    push!(recortes,recorte2);
    push!(recortes,recorte3);
    push!(recortes,recorte4);

    return recortes
end

"""
    @Entradas:
        - Imagen de la que extraer la media para ampliar, de ser posible hasta
            tener un tamaño N x M
        - Tamaño de la imagen "ampliada" a N x M si fue posible
    @Salidas:
        - Imagen original si esta tenía un tamaño superior al especificado
        - Imagen con tamaño N x M, cuyos píxeles añadidos tienen un valor igual
            a la media de todos los píxeles de la imagen original
"""
function ampliarConMediaHastaNM(image, n::Int64, m::Int64)
    image_size = size(image)

    if (image_size[1] > n && image_size(image) > m)
        return image
    end
    
    mean_ = mean(image)



end

"""
    @Entradas:
        - Imagen de la que extraer la media y la desviación típica
        - Clase a la que pertenecería el patrón
    @Salidas:
        - Un vector con la media, desv. y la clase a la que pertenece el patrón
"""
function featureExtraction(image, class::Int64, classes::AbstractArray{<:Any, 1})
    mean_ = mean(image)
    std_ = std(image)
    return [mean_, std_, class]
end

function saveAsData(fileName::String, data, separator::Char = ';')
    open(fileName, "w") do io
        writedlm(io, data, separator)
    end
end


function count(dir::String)
    content = readdir(dir)
    filter!(endswith(".png"), content)
    global num =0
    for i in content
        num +=1
    end
end


#count(img_path)
#println(num)


function count(arr::Vector{Any})
    global num =0
    for i in arr
            num +=1
    end
end


function loadFolderImages(folderName::String)
    images = [];
    files = sort(readdir(folderName), lt=natural)
    filter!(endswith(".png"), files)
    println(files)
    for fileName in files
        println("Loading filename: "*fileName)
        image = load(folderName*"/"*fileName);
            
        # Check that they are color images
        #@assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
        # Add the image to the vector of images
        push!(images, convert(Array{Float64}, gray.(Gray.(image))));
    end;

    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors ||Solo nos interesa lo gris(imageToColorArray.(images), 
    return images;
end;



#---------------------------------------- Funciones P6 -----------------------------------------------------


# Importamos los modelos que vamos a utilizar
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier 


function oneHotEncoding(feature::Matrix{<:Any}, classes::AbstractArray{<:Any, 1})
    unique_classes = unique(classes)

    if size(unique_classes, 1) == 2
        bool_array = Array{Bool, 2}(undef, size(feature, 1), 1)
        bool_array[:, 1] .= (feature.==classes[2])
        return bool_array
    else
        bool_array = falses(size(feature,1), size(unique_classes, 1))
        for i in 1:(size(unique_classes,1 ))
            bool_array[:,i] = convert(Array{Bool,1}, feature.== unique_classes[i])
        end
        return bool_array
    end
end

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    unique_classes = unique(classes)

    if size(unique_classes, 1) == 2
        bool_array = Array{Bool, 2}(undef, size(feature, 1), 1)
        bool_array[:, 1] .= (feature.==classes[2])
        return bool_array
    else
        bool_array = falses(size(feature,1), size(unique_classes, 1))
        for i in 1:(size(unique_classes,1 ))
            bool_array[:,i] = convert(Array{Bool,1}, feature.== unique_classes[i])
        end
        return bool_array
    end

end

#DAVID Tenemos que llamar a la funcion de un solo argumento que ya llama a la funcion con dos argumentos
function oneHotEncoding(feature::AbstractArray{<:Any,1})
    oneHotEncoding(feature, unique(feature))
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, (:,1))
end

#=
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final de la practica 4)
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        # oneHot = Array{Bool,2}(undef, length(targets), numClasses);
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente
=#

function calculateMinMaxNormalizationParameters(input::AbstractArray{<:Real,2})

    min = minimum(input, dims = 1)
    max = maximum(input, dims = 1)
    return(min, max)

end

function calculateZeroMeanNormalizationParameters(input::AbstractArray{<:Real,2})
    
    media = mean(input, dims = 1)
    desv = std(input, dims = 1)
    return(media, desv)
end

"""
    input -> Array bidimensional a normalizar

"""
function normalizeMinMax!(input::AbstractArray{<:Real,2}, parameters::NTuple{2, AbstractArray{<:Real,2}})
    
    input .-= parameters[1]
    input ./= (parameters[2] - parameters[1])

end

normalizeMinMax!(input::AbstractArray{<:Real,2}) = normalizeMinMax!(input, calculateMinMaxNormalizationParameters(input))

function normalizeMinMax!(input::AbstractArray{<:Real,2})
    parameters = calculateMinMaxNormalizationParameters(input)
    normalizeMinMax!(input, parameters)
end

function normalizeMinMax(input::AbstractArray{<:Real,2}, parameters::NTuple{2, AbstractArray{<:Real,2}})
    new_matrix = copy(input)
    normalizeMinMax!(copy, parameters)
    return new_matrix
end

normalizeMinMax(input::AbstractArray{<:Real,2}) = normalizeMinMax(input, calculateMinMaxNormalizationParameters(input))

function normalizeZeroMean!(input::AbstractArray{<:Real,2}, parameters::NTuple{2, AbstractArray{<:Real,2}})
    return (input.-parameters[1])./parameters[2]
end

function normalizeZeroMean!(input::AbstractArray{<:Real,2})
    return normalizeZeroMean!(input, calculateZeroMeanNormalizationParameters(input))
end

# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Comprobamos que efectivamente cada patron solo este clasificado en una clase
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;


function accuracy(targets::AbstractArray{Bool, 1}, outputs::AbstractArray{Bool, 1})
    @assert (size(targets,1)==size(outputs,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"
    
    return Base.count(targets .== outputs)/size(outputs, 1)
end

function accuracy(targets::AbstractArray{Bool, 2}, outputs::AbstractArray{Bool, 2})

    if size(targets, 2) == 1
        accuracy(targets[:,1], outputs[:,1])
    elseif size(targets, 2) > 2
        count = 0
        for i in 1:(size(targets, 1))
            if targets[i, :] == outputs[i, :]
                count += 1
            end
        end
        return count/size(targets, 1)
    end
end

#DAVID Seria para llamarla si no tenemos las salidas clasificadas
#DAVID Pero va a dar igual si la llamamos despues pq seria como llamar dos veces a classifyOutputs que no pasaria nada
function accuracy(targets::AbstractArray{Bool, 1}, outputs::AbstractArray{<:Real,1}, threshold::Real=0.5)
    outputs = outputs .> threshold 
    return accuracy(targets, outputs)
end

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2})
    classComparison = targets .== outputs
    correctClassifications = all(classComparison, dims=2)
    return mean(correctClassifications)
end

#DAVID Creo que vamos cambiar la firma por esta function rna_clasification(topology::AbstractArray{<:Int,1}, targets::AbstractArray{<:Real,2}, outputs::AbstractArray{Bool,2})
#DAVID Se trabaja con features y outputs transpuestos
function rna_clasification(topology::AbstractArray{<:Int,1}, features, outputs)

    ann = Flux.Chain()
    numInputsLayer = size(features, 1)
    
    #DAVID El operador '...' sirve para concatenar mas capas a ann
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ))
        numInputsLayer = numOutputsLayer
    end

    if size(outputs, 1) == 1
        ann = Flux.Chain(ann..., Dense(numInputsLayer, size(outputs,1), σ))
    else 
        ann = Chain(ann..., Dense(numInputsLayer, size(outputs,1)))
        ann = Flux.Chain(ann..., softmax)
    end

    return ann
end

function train(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}, testSet::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}, validationSet::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    trainingInputs = transpose(dataset[1]) # Matriz transpuesta de 4 filas y 150 columnas
    trainingTargets = transpose(dataset[2]) 

    testInputs = transpose(testSet[1])
    testTargets = transpose(testSet[2])

    if (isassigned(validationSet[1], 2))
        validationInputs = transpose(validationSet[1])
        validationTargets = transpose(validationSet[2])
    end


    # Se comprueban que el numero de patrones (filas) coincide tanto en entrenamiento, validacion y test
    @assert (size(trainingInputs, 2) == size(trainingTargets, 2))
    @assert (size(testInputs, 2) == size(testTargets, 2))
    
    if (isassigned(validationSet[1], 2))
        @assert (size(validationInputs, 2) == size(validationTargets, 2))
    end

    # Se comprueba que haya el mismo numero de columnas 
    if (isassigned(validationSet[1], 2))
            @assert (size(trainingInputs, 1) == size(validationInputs,1) == size(testInputs, 1))
            @assert (size(trainingTargets, 1) == size(validationTargets,1) == size(testTargets, 1))
    else 
        @assert (size(trainingInputs, 1)  == size(testInputs, 1))
        @assert (size(trainingTargets, 1) == size(testTargets, 1))
    end

    ann = rna_clasification(topology, trainingInputs, trainingTargets)

    loss(x, y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(ann(x), y) : Flux.Losses.crossentropy(ann(x), y); 

    
    trainingLosses = Float32[]
    trainingAccuracies = Float32[]
    validationLosses = Float32[]
    validationAccuracies = Float32[]
    testLosses = Float32[]
    testAccuracies = Float32[]
    numEpochs = 0;

    function calculateMetrics()

        trainingLoss = loss(trainingInputs, trainingTargets)
        testLoss = loss(testInputs, testTargets)
        
        trainingOutputs = classifyOutputs(ann(trainingInputs))
        testOutputs = classifyOutputs(ann(testInputs))
        
        trainingAccuracy = accuracy(trainingOutputs', trainingTargets')
        testAccuracy = accuracy(testOutputs', testTargets')

        if (isassigned(validationSet[1], 2))
            validationLoss = loss(validationInputs, validationTargets)
            validationOutputs = classifyOutputs(ann(validationInputs))
            validationAccuracy = accuracy(validationOutputs', validationTargets')

            # println("Epoch ", numEpochs, ": Training loss: ", round(trainingLoss, digits = 4), "\t, accuracy: ", round(100*trainingAccuracy, digits=3), "\t % - Validation loss: ", round(validationLoss, digits = 4), "\t, accuracy: ", round(100*validationAccuracy, digits=3), "\t % - Test loss: ", round(testLoss, digits = 4), "\t, accuracy: ", round(100*testAccuracy, digits=3), " %");

            return (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy)
        else
            #println("Epoch ", numEpochs, ": Training loss: ", trainingLoss, ", accuracy: ", round(100*trainingAccuracy, digits=2) , ", % - Test loss: ", testLoss, ", accuracy: ", 100*testAccuracy, " %");

            return (trainingLoss, trainingAccuracy, testLoss, testAccuracy)
        end
        
    end

    (isassigned(validationSet[1], 2)) ?
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics() :
        (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics()
    

    push!(trainingLosses, trainingLoss)
    push!(trainingAccuracies, trainingAccuracy)

    push!(testLosses, testLoss)
    push!(testAccuracies, testAccuracy)

    if (isassigned(validationSet[1], 2))
        push!(validationLosses, validationLoss)
        push!(validationAccuracies, validationAccuracy)
        bestValLoss = validationLoss
    end

    numEpochsVal = 0; 
    bestAnn = deepcopy(ann)

    while (numEpochs < maxEpochs) && (trainingLoss > minLoss) && (numEpochsVal < maxEpochsVal)
        Flux.train!(loss, Flux.params(ann), [(trainingInputs, trainingTargets)], ADAM(learningRate))
        
        numEpochs +=1

        (isassigned(validationSet[1], 2)) ?
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics() :
        (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics()
    
        push!(trainingLosses, trainingLoss)
        push!(trainingAccuracies, trainingAccuracy)

        push!(testLosses, testLoss)
        push!(testAccuracies, testAccuracy)

        if (isassigned(validationSet[1], 2))
            push!(validationLosses, validationLoss)
            push!(validationAccuracies, validationAccuracy)

            if (validationLoss < bestValLoss)
                bestValLoss = validationLoss
                numEpochsVal = 0
                bestAnn = deepcopy(ann)
            else 
                numEpochsVal += 1
            end
        end

    end
        if (isassigned(validationSet[1], 2))
            return (bestAnn, trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies)
        else
            return (ann, trainingLosses, testLosses, trainingAccuracies, testAccuracies)
        end
end

function train(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20, validationSet::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=([undef], [undef]),
    testSet::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=([undef], [undef]))

    train(topology, (dataset[1], reshape(dataset[2], (:,1))), maxEpochs, minLoss, learningRate, maxEpochsVal, (validationSet[1], reshape(validationSet[2], (:,1))),  (testSet[1], reshape(testSet[2], (:,1))))
end


function holdOut(N::Int64, P::Float64)
    @assert ((P>=0.) & (P<=1.))
    n = Int(round(N*P, RoundUp, digits=0))

    full_array = randperm(N)

    test_index = full_array[1:n]
    trainIdex = full_array[n+1:N]

    @assert (size(test_index,1) + size(trainIdex,1) == N)

    return (trainIdex, test_index)
end

function holdOut(N::Int64, Pval::Float64, Ptest::Float64)
    @assert ((Pval>=0.) & (Pval<=1.))
    @assert ((Ptest>=0.) & (Ptest<=1.))
    @assert ((Pval+Ptest)<=1.)

    trainValidationIndex, testIndex = holdOut(N, Ptest) # Se genera el array de indices de entrenamiento y validacion y los indices destinados para test
    
    trainingIndex, validationIndex = holdOut(length(trainValidationIndex), Pval * N / length(trainValidationIndex)) # Se generan nuevos indices para el array de indices

    @assert (size(trainValidationIndex[trainingIndex], 1) + size(trainValidationIndex[validationIndex], 1) + size(testIndex,1) == N )

    return (trainValidationIndex[trainingIndex], trainValidationIndex[validationIndex], testIndex)
end



function crossvalidation(N::Int64, k::Int64)

    array = repeat(1:k, Int64(ceil(N/k)))
    array = array[1:N]
    shuffle!(array)
    return array

end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int64)

    array = zeros(size(targets, 1))

    for i in 1:size(targets, 2)
        nElements = sum(targets[:,i])
        index = crossvalidation(nElements, k)
        array[((i-1)*nElements+1) : (i*nElements)] = index
    end
    return array
end

function crossvalidation(targets::AbstractArray{<:Any, 1}, k::Int64)
    targets = oneHotEncoding(targets)

    return crossvalidation(targets, k)

end

function modelCrossValidation(model::Symbol, parameters::Dict, inputs::Array{Float64,2}, targets::Array{<:Any,2}, k::Int64)

    @assert(size(inputs,1)==length(targets));       # Condición para entradas y salidas deseadas válidas
    @assert((model==:ANN) || (model==:SVM) || (model==:DecisionTree) || (model==:kNN)); # Condición de que debe seguir alguno de los modelos

    testAccuracies = Array{Float64,1}(undef,k);
    testError_rate = Array{Float64,1}(undef,k);
    testRecall = Array{Float64,1}(undef,k);
    testSpeciticity = Array{Float64,1}(undef,k);
    testPrecision = Array{Float64,1}(undef,k);
    testNegative_predictive_value = Array{Float64,1}(undef,k);
    testF1 = Array{Float64,1}(undef,k);

    crossValidationIndex = crossvalidation(size(targets, 1), k)
    # Sacar los campos del modelo
    # kNN: model.n_neighbors, model.metric, model.weights 
    # SVM: model.C, model.support_vectors_, model.support_ 
    #println(keys(model)); 

    if model==:ANN
        # En el caso de entrenar RR.NN.AA., salidas codificadas como en prácticas anteriores. 
        # Parametros:
        #             - arquitectura: num capas ocultas y num de neuronas/capa oculta
        #             - funcion de transferencia en cada capa
        #             - tasa de aprendizaje
        #             - tasa de patrones usados para validacion
        #             - numero maximo de ciclos de entrenamiento
        #             - numero de ciclos sin mejorar el loss de validacion

        classes = unique(targets);
        targets = oneHotEncoding(targets,classes);
    end


    for numFold in 1:k

        # if (model==:SVM) || (model==:DecisionTree) || (model==:kNN)
        if (model!=:ANN)
            # Dividimos entre entrenamiento y test
            trainingInputs = inputs[crossValidationIndex.!=numFold,:];
            testInputs = inputs[crossValidationIndex.==numFold,:];
            trainingTargets = targets[crossValidationIndex.!=numFold];
            testTargets = targets[crossValidationIndex.==numFold];

            if model==:SVM
                model = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
            elseif model==:DecisionTree
                model = DecisionTreeClassifier(max_depth=parameters["maxDepth"], random_state=1)
            elseif model==:kNN
                model = KNeighborsClassifier(parameters["numNeighbors"])
            end

            model = fit!(model, trainingInputs, trainingTargets)

            testOutputs = predict(model, testInputs)

            acc, _, recall, spec, _, _, _, confus = confusionMatrix(testOutputs, testTargets, weighted=true)
            println("Plain confusion: ", confus)
        
        else

            @assert(model==:ANN)
            trainingInputs = inputs[crossValidationIndex.!=numFold,:];
            testInputs = inputs[crossValidationIndex.==numFold,:];
            trainingTargets = targets[crossValidationIndex.!=numFold,:];
            testTargets = targets[crossValidationIndex.==numFold, :];

            testAccuraciesPerRepetition = Array{Float64, 1}(undef, parameters["numExecutions"])
            testSpectPerRepetition = Array{Float64, 1}(undef, parameters["numExecutions"])
            testRecallPerRepetition = Array{Float64, 1}(undef, parameters["numExecutions"])

            for numTraining in 1:parameters["numExecutions"]
                if parameters["validationRatio"] > 0
                    trainingIndex, validationIndex = holdOut(size(trainingInputs, 1), parameters["validationRatio"])
                    ann, _ = train(parameters["topology"], (trainingInputs[trainingIndex, :], trainingTargets[trainingIndex, :]), (testInputs, testTargets),(trainingInputs[validationIndex, :], trainingTargets[validationIndex, :])
                                ;maxEpochs=parameters["maxEpochs"], learningRate=parameters["learningRate"], maxEpochsVal=parameters["maxEpochsVal"])

                else
                    ann, _ = train(parameters["topology"], (trainingInputs, trainingTargets), (testInputs, testTargets), (AbstractMatrix{<:Real}[], AbstractMatrix{Bool}[]);
                    maxEpochs=parameters["maxEpochs"], learningRate=parameters["learningRate"])
                end

                testAccuraciesPerRepetition[numTraining], _, testRecallPerRepetition[numTraining], #=testSpectPerRepetition[numTraining]=#_, _, _, _, confus = confusionMatrix(collect(ann(testInputs')'), testTargets, true)
                println("Plain confusion: ", confus)

            end

            acc = mean(testAccuraciesPerRepetition)
            #spec = mean(testSpectPerRepetition)
            recall = mean(testRecallPerRepetition)

        end

        testAccuracies[numFold] = acc
        #testSpeciticity[numFold] = spec
        testRecall[numFold] = recall

        
        #println("Results in test in fold ", numFold, "/", numFolds, " : accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %")    

    end



    println(model, ": Average test accuracy on ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard desviation of ", 100*std(testAccuracies))
    println(model, ": Average test recall on ", numFolds, "-fold crossvalidation: ", 100*mean(testRecall), ", with a standard desviation of ", 100*std(testRecall))

    return (mean(testAccuracies), std(testAccuracies), mean(testRecall), std(testRecall))

    #return (mean(testAccuracies),std(testAccuracies),mean(testError_rate),std(testError_rate),mean(testRecall),std(testRecall),mean(testSpeciticity),std(testSpeciticity),mean(testPrecision),std(testPrecision),mean(testNegative_predictive_value),std(testNegative_predictive_value),mean(testF1),std(testF1));
end;