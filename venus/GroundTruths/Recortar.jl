#Quitamos el include porque ya se hace en el Main, antes de la llamada "include("Recortar.jl")"
#include("funciones.jl")

# Args

# Número de imágenes a procesar
numberOfImages = 134

for i=1:(size(ARGS,1))
    if ARGS[i] == "-numImages"
        try
            numberOfImages = ARGS[i+1]
        catch
            error("No argument left but expected the number of images to process")
        end
    end
end 
# Debe ser común a todos los PC's
#ccd("Venus_AA/venus/GroundTruths/dataset_etiquetado")

#Guardamos el directorio actual
files_and_dirs = readdir(pattern_path)
println(files_and_dirs)
for f in files_and_dirs
    if filesize(pattern_path*"/"*f) < 3
        rm(pattern_path*"/"*f)
    end
end

#Retiramos del directorio actual los ficheros que no acaben en .lxyr
filter!(endswith(".lxyr"), files_and_dirs)

#Inicializamos la matriz con 1 fila y el numero de atributos de columnas
global dataset = []

#134 = Numero maximo que queremos leer
for count in 1:numberOfImages
    name = "img"*string(count)*".lxyr"
    if isfile(pattern_path*"/"*name) #Checkeamos si existe
        matrix = readdlm(pattern_path*"/"*name)
        push!(dataset, matrix)
    else
        push!(dataset, Float64[])
    end

    count = count + 1
end



size_data = size(dataset, 1)

negative_dataset = []
#Ahora creamos una matriz que guarde lo que fijo son volcanes (1) y los que no (4)
for i=1:size_data
    aux = []
    for j=1:(size(dataset[i], 1))
        if (dataset[i][j, 1] == 4.0)
            push!(aux, dataset[i][j,:])
        end
    end
    push!(negative_dataset, aux)
end

println("Size data negative "*string(size(negative_dataset, 1)))

positive_dataset = []
#Ahora creamos una matriz que guarde lo que fijo son volcanes (1) y los que no (4)
for i=1:size_data
    aux = []
    for j=1:(size(dataset[i], 1))
        if (dataset[i][j, 1] == 1.0)
            push!(aux, dataset[i][j,:])
        end
    end
    push!(positive_dataset, aux)
end



# Tiene todas las imagenes, 1 por fila 
matrix = loadFolderImages(img_path)

save("gray.png", colorview(Gray,matrix[1]))

#display(plot(matrix))

positive_images = []
negative_images = []

for i=1:size_data
    for j=1:(size(negative_dataset[i], 1))
        recorte = negative_dataset[i][j][2:4]
        push!(negative_images, recortar(recorte, matrix[i]))
    end
end

for i=1:size_data
    for j=1:(size(positive_dataset[i], 1))
        recorte = positive_dataset[i][j][2:4]
        push!(positive_images, recortar(recorte, matrix[i]))
    end
end

# Crear el directorio si no existe
if !isdir(hit_path)
    mkdir(hit_path)
end
global cont = 1
for (image) in positive_images
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*".png"
    println("saving \""*name*"\"")
    save(joinpath(hit_path, name), image)
    global cont = cont+1
end

# Crear el directorio si no existe
if !isdir(miss_path)
    mkdir(miss_path)
end
global cont = 1
for (image) in negative_images
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*".png"
    println("saving \""*name*"\"")
    save(joinpath(miss_path, name), image)
    global cont = cont+1
end

#Para recortar las imagenes de la primera aproximacion en 4

positive_images2 = []
negative_images2 = []

for i=1:size_data
    for j=1:(size(negative_dataset[i], 1))
        println(negative_dataset[i][j])
        recorte = negative_dataset[i][j][2:4]
        push!(negative_images2, recortar2(recorte, matrix[i]))
    end
end

for i=1:size_data
    for j=1:(size(positive_dataset[i], 1))
        println(positive_dataset[i][j])
        recorte = positive_dataset[i][j][2:4]
        push!(positive_images2, recortar2(recorte, matrix[i]))
    end
end


# Crear el directorio si no existe
if !isdir(hit_path3)
    mkdir(hit_path3)
end
global cont = 1
for (image) in positive_images2
    for i in image
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(hit_path3, name), i)
        global cont = cont+1
    end
end

# Crear el directorio si no existe
if !isdir(miss_path3)
    mkdir(miss_path3)
end
global cont = 1
for (image) in negative_images2
    for i in image
        # Guardar la imagen en el directorio
        name = "recorte"*string(cont)*"_aprox3.png"
        println("saving \""*name*"\"")
        save(joinpath(miss_path3, name), i)
        global cont = cont+1
    end
end

# Para los recortes a 0.5

K_MULTIPLIER = 0.5

positive_images1 = []
negative_images1 = []

for i=1:size_data
    for j=1:(size(negative_dataset[i], 1))

        recorte = negative_dataset[i][j][2:4]
        push!(negative_images1, recortar(recorte, matrix[i]))
    end
end

for i=1:size_data
    for j=1:(size(positive_dataset[i], 1))

        recorte = positive_dataset[i][j][2:4]
        push!(positive_images1, recortar(recorte, matrix[i]))
    end
end


# Crear el directorio si no existe
if !isdir(hit_path2)
    mkdir(hit_path2)
end
global cont = 1
for (image) in positive_images1
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*"_aprox2.png"
    println("saving \""*name*"\"")
    save(joinpath(hit_path2, name), image)
    global cont = cont+1
end

# Crear el directorio si no existe
if !isdir(miss_path2)
    mkdir(miss_path2)
end
global cont = 1
for (image) in negative_images1
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*"_aprox2.png"
    println("saving \""*name*"\"")
    save(joinpath(miss_path2, name), image)
    global cont = cont+1
end
