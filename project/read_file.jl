using DelimitedFiles


function read_file(
  filename::String,             # path & name of .fvecs file
  n::Int = 11000000,
  d::Int = 28,
  dtype::DataType = Float32
 ) 
  #open(filename, "r") do f
  
   #  while ! eof(f)
    #   s = readline(f)
     #  n += 1
     #end
  #end
  
  #println("Read $n lines")
  
  data = zeros(dtype,d,n)
  
  f = open(filename, "r")     # open input file
  
  i = 1 
  while ! eof(f)
     s = readline(f)
     b = IOBuffer(s)
     
     dataline = readdlm(b, ',', dtype)
     
     for j in 1:d
         data[j,i] = dataline[1,j+1]
     end
     
     i += 1
     
     if i % 110000 == 0
         percnt = i/110000
         println("Processed $i lines so far ( $percnt % )" )
     end
  end
  
  close(f)
  
  return data

end

function store_file(
      filename::String,
      data::AbstractArray,
      append::Bool = false
      )

      if append == false
          f = open(filename, "w")
      else
          f = open(filename, "a")
      end
      
      write(f, data)
      
      close(f)
      
      return
      
end

function load_file(
     filename::String, 
     d::Int = 28, 
     n::Int = 11000000,
     dtype::DataType = Float32
     )

     f = open(filename, "r")
     
     readdata = Array{dtype, 2}(undef, d, n)
     
     read!(f, readdata)
     
     close(f)
     
     return readdata
end

function load_raw_segment_file(
  filename::String, 
  d::Int, 
  indxs::AbstractVector = 1:100000,
  dtype::DataType = Float32
  )
 
  f = open(filename, "r")     # open input file
  
  szel = sizeof( dtype )
  vecsize = d * szel
  
  ntotal = stat( filename ).size ÷ vecsize
  ( stat( filename ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
     
  n = length( indxs )
  
  data = zeros(dtype,d,n)
  
  i = 1 
  while ! eof(f)
     s = readline(f)
     b = IOBuffer(s)
     
     dataline = readdlm(b, ',', dtype)
     
     for j in 1:d
         data[j,i] = dataline[1,j+1]
     end
     
     i += 1
     
     if i % 110000 == 0
         percnt = i/110000
         println("Processed $i lines so far ( $percnt % )" )
     end
  end
  
  close(f)
  
  return data

end

function load_raw_segment_file!(
  filename::String, 
  d::Int, 
  data::AbstractArray, 
  indxs::AbstractVector = 1:100000,
  dtype::DataType = Float32
  )
  
  f = open(filename, "r")     # open input file
  
  szel = sizeof( dtype )
  vecsize = d * szel
  
  ntotal = stat( filename ).size ÷ vecsize
  ( stat( filename ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
  
  n = length( indxs )
  
  if size(data,2) < n 
         error("Incompatible sizes, output vector is larger than the output array")
  end
  
  i = 1 
  while ! eof(f)
     s = readline(f)
     b = IOBuffer(s)
     
     dataline = readdlm(b, ',', dtype)
     
     for j in 1:d
         data[j,i] = dataline[1,j+1]
     end
     
     i += 1
     
     if i % 110000 == 0
         percnt = i/110000
         println("Processed $i lines so far ( $percnt % )" )
     end
  end
  
  close(f)
  
  return data

end

function store_csv_file(
     filename_out::String, 
     d::Int, 
     n::Int, 
     source_file::String, 
     append::Bool = false,
     dtype::DataType = Float32
     )
     
     f = open(source_file, "r")
     if append == true
        o = open(filename_out, "a")
     else
        o = open(filename_out, "w")
     end
     
     szel = sizeof( dtype )
     vecsize =  d * szel
     
     ntotal = stat( source_file ).size ÷ vecsize
     ( stat( source_file ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
     
     dataline = Array{dtype}(undef, 1, d)
     
     while ! eof(f)
        read!(f, dataline)
        
        writedlm(o, dataline, ", ")
     end
     
     close(f)
     close(o)

end 

function store_bin_file(
     filename_out::String, 
     d::Int, 
     n::Int, 
     source_file::String, 
     append::Bool = false,
     dtype::DataType = Float32
     )
     
     f = open(source_file, "r")
     if append == true
        o = open(filename_out, "a")
     else
        o = open(filename_out, "w")
     end
     
     szel = sizeof( dtype )
     vecsize =  d * szel
     
     ntotal = stat( source_file ).size ÷ vecsize
     ( stat( source_file ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
     
     dataline = Array{dtype}(undef, 1, d)
     
     while ! eof(f)
        read!(f, dataline)
        write(o, dataline)
     end
     
     close(f)
     close(o)

end 

function load_bin_segment_file(
     filename::String, 
     d::Int, 
     indxs::AbstractVector = 1:100000,
     dtype::DataType = Float32
     )
     
     #!issorted( indxs ) && @warn("Consider sorting indxs vector for performance.")
     
     f = open(filename, "r")
     
     szel = sizeof( dtype )
     vecsize =  d * szel
     
     ntotal = stat( filename ).size ÷ vecsize
     ( stat( filename ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
     
     n = length( indxs )
     data = Array{dtype}(undef, d, n)
     #off = 4÷szel
     #data = Matrix{Float32}(undef, d + off, n)
     
     if issorted( indxs )
         seek( f, (Vector(indxs)[1] -1)*(vecsize) ) 
         for (col,i) in enumerate( indxs )
           read!( f, view( data, :, col ) )
         end
     else
        @warn("Consider sorting indxs vector for performance.")
        for (col,i) in enumerate( indxs )
           seek( f, (i-1)*(vecsize) ) 
           read!( f, view( data, :, col ) ) 
        end
     end
     
     close(f)
     
     return data

end

function load_bin_segment_file!(
     filename::String, 
     d::Int, 
     data::AbstractArray, 
     indxs::AbstractVector = 1:100000,
     dtype::DataType = Float32
     )
     
     #!issorted( indxs ) && @warn("Consider sorting indxs vector for performance.")
     
     f = open(filename, "r")
     
     szel = sizeof( dtype )
     vecsize =  d * szel
     
     ntotal = stat( filename ).size ÷ vecsize
     ( stat( filename ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
     
     n = length( indxs )
     
     if size(data,2) < n 
         error("Incompatible sizes, output vector is larger than the output array")
     end
     
     #data = Matrix{dtype}(undef, d, n)
     
     #off = 4÷szel
     #data = Matrix{dtype}(undef, d + off, n)
     
     if issorted( indxs )
         seek( f, (Vector(indxs)[1] -1)*(vecsize) ) 
         for (col,i) in enumerate( indxs )
           read!( f, view( data, :, col ) )
         end
     else
        @warn("Consider sorting indxs vector for performance.")
        for (col,i) in enumerate( indxs )
           seek( f, (i-1)*(vecsize) ) 
           read!( f, view( data, :, col ) ) 
        end
     end
     
     close(f)
     
     #@views data_out[:,1:n] .= data[:,1:n]
     
     #data = nothing
end

function load_file_wrapper!(
     filename::String, 
     d::Int, 
     data::AbstractArray, 
     indxs::AbstractVector = 1:100000,
     dtype::DataType = Float32
     )
     
     if filename[findlast(isequal('.'),filename):end] == ".dat"
         load_bin_segment_file!(filename, d, data, indxs, dtype)
     elseif filename[findlast(isequal('.'),filename):end] == ".fvecs"
         load_raw_segment_file!(filename, d, data, indxs, dtype)
     else
         error("File $filename is of an incompatible format. Please use only .fvecs and .dat files only")
     end
     
end

function load_file_wrapper(
     filename::String, 
     d::Int, 
     indxs::AbstractVector = 1:100000,
     dtype::DataType = Float32
     )
     
     if filename[findlast(isequal('.'),filename):end] == ".dat"
         return load_bin_segment_file(filename, d, indxs, dtype)
     elseif filename[findlast(isequal('.'),filename):end] == ".fvecs"
         return load_raw_segment_file(filename, d, indxs, dtype)
     else
         error("File $filename is of an incompatible format. Please use only .fvecs and .dat files only")
     end
     
end
