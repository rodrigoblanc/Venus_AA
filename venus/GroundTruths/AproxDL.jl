using Flux
using Flux.Losses
using Flux: params
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean
using NaturalSort
using Colors
using Images
using Random

function holdOut(N::Int64, P::Float64)
    @assert ((P>=0.) & (P<=1.))
    n = Int(round(N*P, RoundUp, digits=0))

    full_array = randperm(N)

    test_index = full_array[1:n]
    trainIdex = full_array[n+1:N]

    @assert (size(test_index,1) + size(trainIdex,1) == N)

    return (trainIdex, test_index)
end

hit_path = pwd()*"/venus/hit"
miss_path = pwd()*"/venus/miss"

 #=
train_imgs   = load("MNIST.jld2", "train_imgs");
train_labels = load("MNIST.jld2", "train_labels");
test_imgs    = load("MNIST.jld2", "test_imgs");
test_labels  = load("MNIST.jld2", "test_labels");
=#
labels = 0:1; # Las etiquetas
r_size = (28,28)
#Función para cargar las imágenes en memoria:
function loadFolderImages(folderName::String)
    images = [];
    files = sort(readdir(folderName), lt=natural)
    filter!(endswith(".png"), files)
    println(files)
    for fileName in files
        println("Loading filename: "*fileName)
        image = load(folderName*"/"*fileName);

        # Check that they are color images
        #@assert(isa(image, Array{RGBA{Normed{UInt8,8}},2})  isa(image, Array{RGB{Normed{UInt8,8}},2}))
        # Add the image to the vector of images
        push!(images, convert(Array{Float64}, gray.(Gray.(imresize(image,r_size)))));
    end;

    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors Solo nos interesa lo gris(imageToColorArray.(images), 
    return images;
end;
hit = loadFolderImages(hit_path) #Cargamos las imagenes positivas
miss = loadFolderImages(miss_path) #Cargamos las imagenes negativas

n_hit = length(hit)
n_miss = length(miss)

imagenes = vcat(hit, miss)

N = length(imagenes)
P = 0.2  # Proporción para prueba (por ejemplo, 20%)

trainIndex, testIndex = holdOut(N, P)

train_imgs = imagenes[trainIndex]
test_imgs = imagenes[testIndex]

lbl = vcat(fill(0,n_hit), fill(1,n_miss)) #Todas las etiquetas juntas
train_labels = lbl[trainIndex]
test_labels = lbl[testIndex]

# Tanto train_imgs como test_imgs son arrays de arrays bidimensionales (arrays de imagenes), es decir, son del tipo Array{Array{Float32,2},1}
#  Generalmente en Deep Learning los datos estan en tipo Float32 y no Float64, es decir, tienen menos precision
#  Esto se hace, entre otras cosas, porque las tarjetas gráficas (excepto las más recientes) suelen operar con este tipo de dato
#  Si se usa Float64 en lugar de Float32, el sistema irá mucho más lento porque tiene que hacer conversiones de Float64 a Float32

# Para procesar las imagenes con Deep Learning, hay que pasarlas una matriz en formato HWCN
#  Es decir, Height x Width x Channels x N
#  En el caso de esta base de datos
#   Height = 28
#   Width = 28
#   Channels = 1 -> son imagenes en escala de grises
#     Si fuesen en color, Channels = 3 (rojo, verde, azul)
# Esta conversion se puede hacer con la siguiente funcion:
function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(28,28)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,1,i] .= imagenes[i][:,:];
    end;
    return nuevoArray;
end;
train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);

println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))


# Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
# En otro caso, habria que normalizarlas
println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");


# Cuando se tienen tantos patrones de entrenamiento (en este caso 60000),
#  generalmente no se entrena pasando todos los patrones y modificando el error
#  En su lugar, el conjunto de entrenamiento se divide en subconjuntos (batches)
#  y se van aplicando uno a uno

# Hacemos los indices para las particiones
# Cuantos patrones va a tener cada particion
batch_size = 32
# Creamos los indices: partimos el vector 1:N en grupos de batch_size
gruposIndicesBatch = Iterators.partition(1:size(train_imgs,4), batch_size);
println("He creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");


# Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
#  Como primer elemento, las imagenes de ese batch
#     train_imgs[:,:,:,indicesBatch]
#  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
#     Para conseguir estas salidas deseadas, se hace una llamada a la funcion onehotbatch, que realiza un one-hot-encoding de las etiquetas que se le pasen como parametros
#     onehotbatch(train_labels[indicesBatch], labels)
#  Por tanto, cada batch será un par dado por
#     (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels))
# Sólo resta iterar por cada batch para construir el vector de batches
train_set = [ (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];

# Creamos un batch similar, pero con todas las imagenes de test
test_set = (test_imgs, onehotbatch(test_labels, labels));


# Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
train_imgs = nothing;
test_imgs = nothing;
GC.gc(); # Pasar el recolector de basura

funcionTransferenciaCapasConvolucionales = relu;


array_ann = []

#=

#! NO FUNCIONA
push!(array_ann,
    Chain(
        Conv((5, 5), 1=>32, pad=(2,2), relu),
        Conv((3, 3), 32=>64, pad=(1,1), relu),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(128, 2, σ)
    )
)

# TODO FUNCIONA PERO NO ACABA
push!(array_ann,
    Chain(
        Conv((3, 3), 1=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 128=>256, pad=(1,1), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(2304, 2, σ)
    )
)

# TODO FUNCIONA PERO NO ACABA
push!(array_ann,
    Chain(
        Conv((5, 5), 1=>32, pad=(2,2), relu),
        MaxPool((2,2)),
        Conv((5, 5), 32=>64, pad=(2,2), relu),
        MaxPool((2,2)),
        Conv((5, 5), 64=>128, pad=(2,2), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(1152, 2, σ)
    )
)

1

push!(array_ann,
    Chain(
    #Parece que Dani parte de 36 imágenes.
        # Primera capa: convolucion, que opera sobre una imagen 28x28
        # Argumentos:
        #  (3, 3): Tamaño del filtro de convolucion
        #  1=>16:
        #   1 canal de entrada: una imagen (matriz) de entradas
        #      En este caso, hay un canal de entrada porque es una imagen en escala de grises
        #      Si fuese, por ejemplo, una imagen en RGB, serian 3 canales de entrada
        #   16 canales de salida: se generan 16 filtros
        #  Es decir, se generan 16 imagenes a partir de la imagen original con filtros 3x3
        # Entradas a esta capa: matriz 4D de dimension 28 x 28 x 1canal    x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension 28 x 28 x 16canales x <numPatrones>
        Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

        # Capa maxpool: es una funcion
        # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes 28x28 a 14x14
        # Entradas a esta capa: matriz 4D de dimension 28 x 28 x 16canales x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension 14 x 14 x 16canales x <numPatrones>
        MaxPool((2,2)),

        # Tercera capa: segunda convolucion: Le llegan 16 imagenes de tamaño 14x14
        #  16=>32:
        #   16 canales de entrada: 16 imagenes (matrices) de entradas
        #   32 canales de salida: se generan 32 filtros (cada uno toma entradas de 16 imagenes)
        #  Es decir, se generan 32 imagenes a partir de las 16 imagenes de entrada con filtros 3x3
        # Entradas a esta capa: matriz 4D de dimension 14 x 14 x 16canales x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension 14 x 14 x 32canales x <numPatrones>
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

        # Capa maxpool: es una funcion
        # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes 14x14 a 7x7
        # Entradas a esta capa: matriz 4D de dimension 14 x 14 x 32canales x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension  7 x  7 x 32canales x <numPatrones>
        MaxPool((2,2)),

        # Tercera convolucion, le llegan 32 imagenes de tamaño 7x7
        #  32=>32:
        #   32 canales de entrada: 32 imagenes (matrices) de entradas
        #   32 canales de salida: se generan 32 filtros (cada uno toma entradas de 32 imagenes)
        #  Es decir, se generan 32 imagenes a partir de las 32 imagenes de entrada con filtros 3x3
        # Entradas a esta capa: matriz 4D de dimension 7 x 7 x 32canales x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension 7 x 7 x 32canales x <numPatrones>
        Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

        # Capa maxpool: es una funcion
        # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes 7x7 a 3x3
        # Entradas a esta capa: matriz 4D de dimension 7 x 7 x 32canales x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension 3 x 3 x 32canales x <numPatrones>
        MaxPool((2,2)),

        # Cambia el tamaño del tensot 3D en uno 2D
        #  Pasa matrices H x W x C x N a matrices H*W*C x N
        #  Es decir, cada patron de tamaño 3 x 3 x 32 lo convierte en un array de longitud 3*3*32
        # Entradas a esta capa: matriz 4D de dimension 3 x 3 x 32canales x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension 288 x <numPatrones>
        x -> reshape(x, :, size(x, 4)),

        # Capa totalmente conectada
        #  Como una capa oculta de un perceptron multicapa "clasico"
        #  Parametros: numero de entradas (288) y numero de salidas (10)
        #   Se toman 10 salidas porque tenemos 10 clases (numeros de 0 a 9)
        # Entradas a esta capa: matriz 4D de dimension 288 x <numPatrones>
        # Salidas de esta capa: matriz 4D de dimension  10 x <numPatrones>
        Dense(288, 2, σ)

        # Finalmente, capa softmax
        #  Toma las salidas de la capa anterior y aplica la funcion softmax de tal manera
        #   que las 10 salidas sean valores entre 0 y 1 con las probabilidades de que un patron
        #   sea de una clase determinada (es decir, las probabilidades de que sea un digito determinado)
        #  Y, ademas, la suma de estas probabilidades sea igual a 1

        #softmax -> Lo quitamos porque solo tenemos 2 salidas.

        # Cuidado: En esta RNA se aplica la funcion softmax al final porque se tienen varias clases
        # Si sólo se tuviesen 2 clases, solo se tiene una salida, y no seria necesario utilizar la funcion softmax
        #  En su lugar, la capa totalmente conectada tendria como funcion de transferencia una sigmoidal (devuelve valores entre 0 y 1)
        #  Es decir, no habria capa softmax, y la capa totalmente conectada seria la ultima, y seria Dense(288, 10, σ)

    )
)

2
#FUNCIONA
push!(array_ann,
    Chain(
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 16=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(1152, 2, σ)
    )
)


3

#FUNCIONA PERO MAL RESULTADO
push!(array_ann,
    Chain(
        Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(64, 2, σ)
    )
)

4

#FUNCIONA
push!(array_ann,
    Chain(
        Conv((1, 1), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((4,4)),
        Conv((2, 2), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(512, 2, σ),
    )
)

5

#FUNCIONA
push!(array_ann,
    Chain(
        Conv((1, 1), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((4,4)),
        Conv((2, 2), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((4, 4), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((3,3)),
        Conv((2, 2), 32=>64, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(64, 2, σ),
    )
)

=#

#FUNCIONA
push!(array_ann,
    Chain(
        Conv((1, 1), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((4,4)),
        Conv((2, 2), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((4,4)),
        Conv((2, 2), 32=>128, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(128, 2, σ),
    )
)


    # Vamos a probar la RNA capa por capa y poner algunos datos de cada capa
    # Usaremos como entrada varios patrones de un batch
    numBatchCoger = 1; numImagenEnEseBatch = [12, 6];
    # Para coger esos patrones de ese batch:
    #  train_set es un array de tuplas (una tupla por batch), donde, en cada tupla, el primer elemento son las entradas y el segundo las salidas deseadas
    #  Por tanto:
    #   train_set[numBatchCoger] -> La tupla del batch seleccionado
    #   train_set[numBatchCoger][1] -> El primer elemento de esa tupla, es decir, las entradas de ese batch
    #   train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch] -> Los patrones seleccionados de las entradas de ese batch
    entradaCapa = train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch];

function train_ann(ann)

    numCapas = length(params(ann));
    println("La RNA tiene ", numCapas, " capas:");
    for numCapa in 1:numCapas
        println("   Capa ", numCapa, ": ", ann[numCapa]);
        # Le pasamos la entrada a esta capa
        global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
        capa = ann[numCapa];
        salidaCapa = capa(entradaCapa);
        println("      La salida de esta capa tiene dimension ", size(salidaCapa));
        entradaCapa = salidaCapa;
    end

    # Sin embargo, para aplicar un patron no hace falta hacer todo eso.
    #  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
    ann(train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch]);




    # Definimos la funcion de loss de forma similar a las prácticas de la asignatura
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
    accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
    # Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


    # Mostramos la precision antes de comenzar el entrenamiento:
    #  train_set es un array de batches
    #  accuracy recibe como parametro un batch
    #  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
    #   y devuelve un array con los resultados
    #  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
    #   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
    println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy.(train_set)), " %");


    # Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
    opt = ADAM(0.001);


    println("Comenzando entrenamiento...")
    global mejorPrecision = -Inf;
    global criterioFin = false;
    global numCiclo = 0;
    global numCicloUltimaMejora = 0;
    global mejorModelo = nothing;

    while (!criterioFin)

        # Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
        global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

        # Se entrena un ciclo
        Flux.train!(loss, params(ann), train_set, opt);

        global numCiclo += 1;

        # Se calcula la precision en el conjunto de entrenamiento:
        global precisionEntrenamiento = mean(accuracy.(train_set));
        println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            global mejorPrecision = precisionEntrenamiento;
            global precisionTest = accuracy(test_set);
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            global mejorModelo = deepcopy(ann);
            global numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0
            println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
            global numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
            global criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            global criterioFin = true;
        end
    end
end

for ann in array_ann
    train_ann(ann) 
end
