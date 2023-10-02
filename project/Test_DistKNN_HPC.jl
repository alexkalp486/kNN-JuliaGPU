ENV["GKSwstype"] = "nul"
using FLANN, BenchmarkTools, Statistics, Distributed, DelimitedFiles, CUDA, Plots, ColorSchemes
using Plots.PlotMeasures

using SharedArrays, Printf

const IN_SLURM = "SLURM_JOBID" in keys(ENV)

# load packages
IN_SLURM && using ClusterManagers

# Here we create our parallel julia processes
if IN_SLURM
    pids = addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
    print("\n")
else
    pids = addprocs()
end

# See ids of our workers. Should be same length as SLURM_NTASKS
# The output of this `println` command will appear in the
# SLURM output file julia_in_parallel.output
println(workers())

@sync begin
    @everywhere include("read_file.jl")
end
@sync begin
    @everywhere include("DistKNN.jl")
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~              Convenient consts
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#filename_input = "./KNNdata/HIGGS_sample.dat"
filename_input = "./KNNdata/HIGGS_big_sample.dat"

file_d_indxs = "./KNNdata/Higgs_d_indxs.dat"
file_d_dists = "./KNNdata/Higgs_d_dists.dat"

file_a_indxs = "./KNNdata/Higgs_a_indxs.dat"
file_a_dists = "./KNNdata/Higgs_a_dists.dat"

file_g_indxs = "./KNNdata/Higgs_g_indxs.dat"
file_g_dists = "./KNNdata/Higgs_g_dists.dat"

file_c_indxs = "./KNNdata/Higgs_c_indxs.dat"
file_c_dists = "./KNNdata/Higgs_c_dists.dat"

file_p_indxs = "./KNNdata/Higgs_p_indxs.dat"
file_p_dists = "./KNNdata/Higgs_p_dists.dat"

file_rpg_indxs = "./KNNdata/Higgs_rpg_indxs.dat"
file_rpg_dists = "./KNNdata/Higgs_rpg_dists.dat"

file_rpc_indxs = "./KNNdata/Higgs_rpc_indxs.dat"
file_rpc_dists = "./KNNdata/Higgs_rpc_dists.dat"

file_cg_indxs = "./KNNdata/Higgs_cg_indxs.dat"
file_cg_dists = "./KNNdata/Higgs_cg_dists.dat"

file_cc_indxs = "./KNNdata/Higgs_cc_indxs.dat"
file_cc_dists = "./KNNdata/Higgs_cc_dists.dat"

file_nn_indxs = "./KNNdata/Higgs_nn_indxs.dat"
file_nn_dists = "./KNNdata/Higgs_nn_dists.dat"

file_fs_indxs = "./KNNdata/Higgs_fs_indxs.dat"
file_fs_dists = "./KNNdata/Higgs_fs_dists.dat"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~                 Utilities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


function calcAccStrict(
      idxs_bf::AbstractArray, 
      idxs_temp::AbstractArray
      )

      if size(idxs_bf, 2) != size(idxs_temp, 2)
         error("Mismatching dimensions of the input arrays of calcAcc")
      end

      accRes = 100.0*(sum(idxs_bf .== idxs_temp))/(size(idxs_bf,1)*size(idxs_bf,2))
      
      return accRes
end

function calcAcc(
      idxs_bf::AbstractArray, 
      idxs_temp::AbstractArray
      )

      if size(idxs_bf, 2) != size(idxs_temp, 2)
         error("Mismatching dimensions of the input arrays of calcAcc")
      end
      
      commonIndxs = 0
      
      @inbounds for i in 1:size(idxs_bf,2)
          commonIndxs = commonIndxs + length(findall(in(idxs_temp[:,i]), idxs_bf[:,i]))
      end
      
      #commonIndxs = length(findall(in(idxs_temp), idxs_bf))
      accRes = 100.0*(commonIndxs)/(size(idxs_bf,1)*size(idxs_bf,2))
      
      return accRes
end


function calcAcc_bin(
      filename_bf::String, 
      filename_temp::String,
      k::Int = 150,
      dtype::DataType = Int32
      )
      
      if isfile(filename_bf) == false
         error("Input file @filename_bf doesn't exist. Check path or filename")
      end
      if isfile(filename_temp) == false
         error("Input file @filename_temp doesn't exist. Check path or filename")
      end
      
      szel = sizeof( dtype )
      vecsize = k * szel
  
      n_bf = stat( filename_bf ).size ÷ vecsize
      ( stat( filename_bf ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      n_temp = stat( filename_temp ).size ÷ vecsize
      ( stat(filename_temp).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      if n_bf != n_temp
         error("The files @filename_bf and @filename_temp have different sizes")
      end
      
      readdata_bf = Array{dtype, 2}(undef, k, n_bf)
      readdata_temp = Array{dtype, 2}(undef, k, n_temp)
      
      f_bf = open(filename_bf, "r")
      f_temp = open(filename_temp, "r")
      
      read!(f_bf, readdata_bf)
      read!(f_temp, readdata_temp)
      
      close(f_bf)
      close(f_temp)
      
      commonIndxs = 0
      
      for i in 1:n_bf
         commonIndxs = commonIndxs + length(findall(in(readdata_temp[2:end,i]), readdata_bf[2:end,i]))
      end
      
      accRes = 100.0*(commonIndxs)/(k * n_bf)

      return accRes
end

function calcRecall(
      idxs_bf::AbstractArray, 
      idxs_temp::AbstractArray
      )

      if size(idxs_bf, 2) != size(idxs_temp, 2)
         error("Mismatching dimensions of the input arrays of calcAcc")
      end
      
      totalElements = (size(idxs_bf,1)*size(idxs_bf,2))
      
      commonIndxs = 0
      
      @inbounds for i in 1:size(idxs_bf,2)
          commonIndxs = commonIndxs + length(findall(in(idxs_temp[:,i]), idxs_bf[:,i]))
      end
      
      truePositives = commonIndxs
      falseNegatives = totalElements - truePositives
      
      recRes = (truePositives)/(truePositives + falseNegatives)
      
      return recRes
end

function calcRecall_bin(
      filename_bf::String, 
      filename_temp::String,
      k::Int = 150,
      dtype::DataType = Int32
      )
      
      if isfile(filename_bf) == false
         error("Input file @filename_bf doesn't exist. Check path or filename")
      end
      if isfile(filename_temp) == false
         error("Input file @filename_temp doesn't exist. Check path or filename")
      end
      
      szel = sizeof( dtype )
      vecsize = k * szel
  
      n_bf = stat( filename_bf ).size ÷ vecsize
      ( stat( filename_bf ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      n_temp = stat( filename_temp ).size ÷ vecsize
      ( stat(filename_temp).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      if n_bf != n_temp
         error("The files @filename_bf and @filename_temp have different sizes")
      end
      
      readdata_bf = Array{dtype, 2}(undef, k, n_bf)
      readdata_temp = Array{dtype, 2}(undef, k, n_temp)
      
      f_bf = open(filename_bf, "r")
      f_temp = open(filename_temp, "r")
      
      read!(f_bf, readdata_bf)
      read!(f_temp, readdata_temp)
      
      close(f_bf)
      close(f_temp)
      
      totalElements = k * n_bf
      
      commonIndxs = 0
      
      for i in 1:n_bf
         commonIndxs = commonIndxs + length(findall(in(readdata_temp[2:end,i]), readdata_bf[2:end,i]))
      end
      
      truePositives = commonIndxs
      falseNegatives = totalElements - truePositives
      
      recRes = (truePositives)/(truePositives + falseNegatives)

      return recRes
end

function calcBest(
      dists_bf::AbstractArray, 
      dists_temp::AbstractArray
      )

      if size(dists_bf, 2) != size(dists_temp, 2)
         error("Mismatching sizes of the input arrays of calcBest")
      end
      if size(dists_bf, 1) != size(dists_temp, 1)
         error("Mismatching dimensions of the input arrays of calcBest")
      end
      
      k = size(dists_bf, 1)
      n_bf = size(dists_bf, 2)
      
      best_bf = 0
      best_temp = 0
      best_tie = 0
      
      for i in 1:n_bf
         for j in 2:k
             if isapprox(dists_bf[j,i], dists_temp[j,i]; atol = 1e-5)
                best_tie += 1
             elseif dists_bf[j,i] < dists_temp[j,i]
                best_bf += 1
             else 
                best_temp += 1
             end
         end
      end

      winPer_bf = 100.0*(best_bf)/((k-1) * n_bf)
      winPer_temp = 100.0*(best_temp)/((k-1) * n_bf)
      winPer_tie = 100.0*(best_tie)/((k-1) * n_bf)

      println("k = $k, n_bf = $n_bf, best_bf = $best_bf, best_temp = $best_temp, best_tie = $best_tie")      
      
      return (winPer_bf, winPer_temp, winPer_tie)
end

function calcBest_bin(
      filename_bf::String, 
      filename_temp::String,
      k::Int = 150,
      dtype::DataType = Float32
      )

      if isfile(filename_bf) == false
         error("Input file @filename_bf doesn't exist. Check path or filename")
      end
      if isfile(filename_temp) == false
         error("Input file @filename_temp doesn't exist. Check path or filename")
      end
      
      szel = sizeof( dtype )
      vecsize = k * szel
  
      n_bf = stat( filename_bf ).size ÷ vecsize
      ( stat( filename_bf ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      n_temp = stat( filename_temp ).size ÷ vecsize
      ( stat(filename_temp).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      if n_bf != n_temp
         error("The files @filename_bf and @filename_temp have different sizes")
      end
      
      readdata_bf = Array{dtype, 2}(undef, k, n_bf)
      readdata_temp = Array{dtype, 2}(undef, k, n_temp)
      
      f_bf = open(filename_bf, "r")
      f_temp = open(filename_temp, "r")
      
      read!(f_bf, readdata_bf)
      read!(f_temp, readdata_temp)
      
      close(f_bf)
      close(f_temp)
      
      best_bf = 0
      best_temp = 0
      best_tie = 0
      
      for i in 1:n_bf
         for j in 2:k
             if isapprox(readdata_bf[j,i], readdata_temp[j,i]; atol = 1e-5)
                best_tie += 1
             elseif readdata_bf[j,i] < readdata_temp[j,i]
                best_bf += 1
             else
                best_temp += 1
             end
         end
      end

      winPer_bf = 100.0*(best_bf)/(k * n_bf)
      winPer_temp = 100.0*(best_temp)/(k * n_bf)
      winPer_tie = 100.0*(best_tie)/(k * n_bf) 
      
      return (winPer_bf, winPer_temp, winPer_tie)
end

function calcErr(
      dists_bf::AbstractArray, 
      dists_temp::AbstractArray
      )

      if size(dists_bf, 2) != size(dists_temp, 2)
         error("Mismatching dimensions of the input arrays of calcErr")
      end
      
      distErr = zeros(Float64, size(dists_bf, 2))
      
      @inbounds for i in 1:size(dists_bf,2)
          sum_bf = 0.0
          sum_temp = 0.0
          
          for j in 1:size(dists_bf,1)
             sum_bf += dists_bf[j,i]
             sum_temp += dists_temp[j,i]
          end 
          distErr[i] = sum_temp - sum_bf
      end
      
      err = std(distErr)
      
      return err
end

function calcErr_bin(
      filename_bf::String, 
      filename_temp::String,
      k::Int = 150,
      dtype::DataType = Float32
      )

      if isfile(filename_bf) == false
         error("Input file @filename_bf doesn't exist. Check path or filename")
      end
      if isfile(filename_temp) == false
         error("Input file @filename_temp doesn't exist. Check path or filename")
      end
      
      szel = sizeof( dtype )
      vecsize = k * szel
  
      n_bf = stat( filename_bf ).size ÷ vecsize
      ( stat( filename_bf ).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      n_temp = stat( filename_temp ).size ÷ vecsize
      ( stat(filename_temp).size % vecsize ) != 0 && error("Incompatible sizes, check files.")
      
      if n_bf != n_temp
         error("The files @filename_bf and @filename_temp have different sizes")
      end
      
      f_bf = open(filename_bf, "r")
      f_temp = open(filename_temp, "r")
      
      distErr = zeros(Float64, n_bf)
      readdata_bf = Array{dtype, 2}(undef, k, n_bf)
      readdata_temp = Array{dtype, 2}(undef, k, n_temp)
      
      read!(f_bf, readdata_bf)
      read!(f_temp, readdata_temp)
      
      close(f_bf)
      close(f_temp)
      
      for i in 1:n_bf
        sum_bf = sum(readdata_bf[2:end,i])
        sum_temp = sum(readdata_temp[2:end,i])
       
        distErr[i] = sum_temp - sum_bf
      end
      
      err = std(distErr)
      
      return err
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~                 Testing code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function batchWarmup(
      filename_input_C::String,
      filename_input_Q::String,
      k::Int = 150, 
      d::Int = 28,
      r::Int = 3,
      P::Int = 5,
      C_size:: Int = 10000,
      Q_size::Int = 5000,
      in_memory::Bool = true
      )

      nPoints = 4
      nAlg = 11
      
      resultsAcc = zeros(Float64, nPoints,nAlg)
      resultsErr = zeros(Float64, nPoints,nAlg)
      resultsRec = zeros(Float64, nPoints,nAlg)
      resultsTime = zeros(Float64, nPoints,nAlg)
      resultsQps = zeros(Float64, nPoints,nAlg)
      
      resultsBest = zeros(Float64, nPoints,nAlg)
      resultsBest2 = zeros(Float64, nPoints,nAlg)
      resultsBest3 = zeros(Float64, nPoints,nAlg)
      resultsBest4 = zeros(Float64, nPoints,nAlg)
      
      pointsC = 1000
      if filename_input_C == filename_input_Q
         pointsQ = 1000
      else
         pointsQ = 250
      end
      
      count_p = 1
      
      if C_size > pointsC
         C_size = pointsC
      end
      
      if Q_size > pointsQ
         Q_size = pointsQ
      end
      
      if k > pointsQ
         k = pointsQ
      end
      
      cid = 1

      println("=========================================")
      println("\t Starting loop for $(pointsC) Corpus points and $(pointsQ) Query points")
      println("=========================================")
      
      println("\nCalculating cpu results")
     if in_memory == false

          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_c_indxs,file_c_dists,k,d, pointsC,pointsQ,C_size,Q_size,3,in_memory)
          
          resultsAcc[count_p,4] = 100.0
          resultsErr[count_p,4] = 0.0
          resultsRec[count_p,4] = 1.0
          resultsTime[count_p,4] = b_t
          resultsQps[count_p,4] = pointsQ/b_t
      else
          indxs_c = zeros(Int32,k,pointsQ)
          dists_c = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_c, dists_c = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,3,in_memory)
          
          resultsAcc[count_p,4] = 100.0
          resultsErr[count_p,4] = 0.0
          resultsRec[count_p,4] = 1.0
          resultsTime[count_p,4] = b_t
          resultsQps[count_p,4] = pointsQ/b_t
      end
      
     println("$(resultsTime[count_p,4]) secs")
     println("$(resultsAcc[count_p,4]) %")
     println("$(resultsRec[count_p,4]) recall")
     
     @everywhere GC.gc(true)
     
      println("\nCalculating FLANN brute force results")
      if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_d_indxs,file_d_dists,k,d, pointsC,pointsQ,C_size,Q_size,0,in_memory)
          
          resultsAcc[count_p,1] = calcAcc_bin(file_c_indxs,file_d_indxs,k,Int32)
          resultsErr[count_p,1] = calcErr_bin(file_c_dists,file_d_dists,k,Float32)
          resultsRec[count_p,1] = calcRecall_bin(file_c_indxs,file_d_indxs,k,Int32)
          resultsTime[count_p,1] = b_t
          resultsQps[count_p,1] = pointsQ/b_t
      else
          indxs_d = zeros(Int32,k,pointsQ)
          dists_d = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_d, dists_d = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,0,in_memory)
         
          resultsAcc[count_p,1] = calcAcc(indxs_c,indxs_d)
          resultsErr[count_p,1] = calcErr(dists_c,dists_d)
          resultsRec[count_p,1] = calcRecall(indxs_c,indxs_d)
          resultsTime[count_p,1] = b_t
          resultsQps[count_p,1] = pointsQ/b_t
      end
      

      println("$(resultsTime[count_p,1]) secs")
      println("$(resultsAcc[count_p,1]) %")
      println("$(resultsRec[count_p,1]) recall")
  
  
     @everywhere GC.gc(true)
  
     println("\nCalculating FLANN approximated results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_a_indxs,file_a_dists,k,d, pointsC,pointsQ,C_size,Q_size,1,in_memory)
          
          resultsAcc[count_p,2] = calcAcc_bin(file_c_indxs,file_a_indxs,k,Int32)
          resultsErr[count_p,2] = calcErr_bin(file_c_dists,file_a_dists,k,Float32)
          resultsRec[count_p,2] = calcRecall_bin(file_c_indxs,file_a_indxs,k,Int32)
          resultsTime[count_p,2] = b_t
          resultsQps[count_p,2] = pointsQ/b_t
      else
          indxs_a = zeros(Int32,k,pointsQ)
          dists_a = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_a, dists_a = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,1,in_memory)
         
          resultsAcc[count_p,2] = calcAcc(indxs_c,indxs_a)
          resultsErr[count_p,2] = calcErr(dists_c,dists_a)
          resultsRec[count_p,2] = calcRecall(indxs_c,indxs_a)
          resultsTime[count_p,2] = b_t
          resultsQps[count_p,2] = pointsQ/b_t
      end
      
     println("$(resultsTime[count_p,2]) secs")
     println("$(resultsAcc[count_p,2]) %")
     println("$(resultsRec[count_p,2]) recall")
  
  
     @everywhere GC.gc(true)
  
     println("\nCalculating brute force gpu results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_g_indxs,file_g_dists,k,d, pointsC,pointsQ,C_size,Q_size,2,in_memory)
          
          resultsAcc[count_p,3] = calcAcc_bin(file_c_indxs,file_g_indxs,k,Int32)
          resultsErr[count_p,3] = calcErr_bin(file_c_dists,file_g_dists,k,Float32)
          resultsRec[count_p,3] = calcRecall_bin(file_c_indxs,file_g_indxs,k,Int32)
          resultsTime[count_p,3] = b_t
          resultsQps[count_p,3] = pointsQ/b_t
      else
          indxs_g = zeros(Int32,k,pointsQ)
          dists_g = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_g, dists_g = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,2,in_memory)
          
          resultsAcc[count_p,3] = calcAcc(indxs_c,indxs_g)
          resultsErr[count_p,3] = calcErr(dists_c,dists_g)
          resultsRec[count_p,3] = calcRecall(indxs_c,indxs_g)
          resultsTime[count_p,3] = b_t
          resultsQps[count_p,3] = pointsQ/b_t
      end
      
      println("$(resultsTime[count_p,3]) secs")
      println("$(resultsAcc[count_p,3]) %")
      println("$(resultsRec[count_p,3]) recall")
      
     @everywhere CUDA.reclaim()
     @everywhere GC.gc(true)
     
      println("\nCalculating Random Projection GPU results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_rpg_indxs,file_rpg_dists,k,d, pointsC,pointsQ,C_size,Q_size,6,in_memory)
          
          resultsAcc[count_p,5] = calcAcc_bin(file_c_indxs,file_rpg_indxs,k,Int32)
          resultsErr[count_p,5] = calcErr_bin(file_c_dists,file_rpg_dists,k,Float32)
          resultsRec[count_p,5] = calcRecall_bin(file_c_indxs,file_rpg_indxs,k,Int32)
          resultsTime[count_p,5] = b_t
          resultsQps[count_p,5] = pointsQ/b_t
      else
          indxs_rpg = zeros(Int32,k,pointsQ)
          dists_rpg = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_rpg, dists_rpg = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,6,in_memory,r,P)
          
          resultsAcc[count_p,5] = calcAcc(indxs_c,indxs_rpg)
          resultsErr[count_p,5] = calcErr(dists_c,dists_rpg)
          resultsRec[count_p,5] = calcRecall(indxs_c,indxs_rpg)
          resultsTime[count_p,5] = b_t
          resultsQps[count_p,5] = pointsQ/b_t
      end
      
      println("$(resultsTime[count_p,5]) secs")
      println("$(resultsAcc[count_p,5]) %")
      println("$(resultsRec[count_p,5]) recall")
      
     @everywhere CUDA.reclaim()
     @everywhere GC.gc(true)
     
     
     println("\nCalculating Random Projection CPU results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_rpc_indxs,file_rpc_dists,k,d, pointsC,pointsQ,C_size,Q_size,5,in_memory,r,P)
          
          resultsAcc[count_p,6] = calcAcc_bin(file_c_indxs,file_rpc_indxs,k,Int32)
          resultsErr[count_p,6] = calcErr_bin(file_c_dists,file_rpc_dists,k,Float32)
          resultsRec[count_p,6] = calcRecall_bin(file_c_indxs,file_rpc_indxs,k,Int32)
          resultsTime[count_p,6] = b_t
          resultsQps[count_p,6] = pointsQ/b_t
      else
          indxs_rpc = zeros(Int32,k,pointsQ)
          dists_rpc = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_rpc, dists_rpc = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,5,in_memory,r,P)
          
          resultsAcc[count_p,6] = calcAcc(indxs_c,indxs_rpc)
          resultsErr[count_p,6] = calcErr(dists_c,dists_rpc)
          resultsRec[count_p,6] = calcRecall(indxs_c,indxs_rpc)
          resultsTime[count_p,6] = b_t
          resultsQps[count_p,6] = pointsQ/b_t
      end
      
      println("$(resultsTime[count_p,6]) secs")
      println("$(resultsAcc[count_p,6]) %")
      println("$(resultsRec[count_p,6]) recall")
      
      
     println("\nCalculating Cluster TI GPU results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_cg_indxs,file_cg_dists,k,d, pointsC,pointsQ,C_size,Q_size,8,in_memory)
          
          resultsAcc[count_p,7] = calcAcc_bin(file_c_indxs,file_cg_indxs,k,Int32)
          resultsErr[count_p,7] = calcErr_bin(file_c_dists,file_cg_dists,k,Float32)
          resultsRec[count_p,7] = calcRecall_bin(file_c_indxs,file_cg_indxs,k,Int32)
          resultsTime[count_p,7] = b_t
          resultsQps[count_p,7] = pointsQ/b_t
      else
          indxs_cg = zeros(Int32,k,pointsQ)
          dists_cg = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_cg, dists_cg = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,8,in_memory)
         
          resultsAcc[count_p,7] = calcAcc(indxs_c,indxs_cg)
          resultsErr[count_p,7] = calcErr(dists_c,dists_cg)
          resultsRec[count_p,7] = calcRecall(indxs_c,indxs_cg)
          resultsTime[count_p,7] = b_t
          resultsQps[count_p,7] = pointsQ/b_t
      end
      
      println("$(resultsTime[count_p,7]) secs")
      println("$(resultsAcc[count_p,7]) %")
      println("$(resultsRec[count_p,7]) recall")
     
     @everywhere CUDA.reclaim()
     @everywhere GC.gc(true)          
      
      
     println("\nCalculating Cluster TI CPU results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_cc_indxs,file_cc_dists,k,d, pointsC,pointsQ,C_size,Q_size,7,in_memory)
          
          resultsAcc[count_p,8] = calcAcc_bin(file_c_indxs,file_cc_indxs,k,Int32)
          resultsErr[count_p,8] = calcErr_bin(file_c_dists,file_cc_dists,k,Float32)
          resultsRec[count_p,8] = calcRecall_bin(file_c_indxs,file_cc_indxs,k,Int32)
          resultsTime[count_p,8] = b_t
          resultsQps[count_p,8] = pointsQ/b_t
      else
          indxs_cc = zeros(Int32,k,pointsQ)
          dists_cc = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_cc, dists_cc = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,7,in_memory)
          
          resultsAcc[count_p,8] = calcAcc(indxs_c,indxs_cc)
          resultsErr[count_p,8] = calcErr(dists_c,dists_cc)
          resultsRec[count_p,8] = calcRecall(indxs_c,indxs_cc)
          resultsTime[count_p,8] = b_t
          resultsQps[count_p,8] = pointsQ/b_t
      end
      
      println("$(resultsTime[count_p,8]) secs")
      println("$(resultsAcc[count_p,8]) %")
      println("$(resultsRec[count_p,8]) recall")

     @everywhere GC.gc(true)           
     
     println("\nCalculating ParallelKNN (gpu) results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_p_indxs,file_p_dists,k,d, pointsC,pointsQ,C_size,Q_size,4,in_memory)
          
          resultsAcc[count_p,9] = calcAcc_bin(file_c_indxs,file_p_indxs,k,Int32)
          resultsErr[count_p,9] = calcErr_bin(file_c_dists,file_p_dists,k,Float32)
          resultsRec[count_p,9] = calcRecall_bin(file_c_indxs,file_p_indxs,k,Int32)
          resultsTime[count_p,9] = b_t
          resultsQps[count_p,9] = pointsQ/b_t
      else
          indxs_p = zeros(Int32,k,pointsQ)
          dists_p = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_p, dists_p = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,4,in_memory)
          
          resultsAcc[count_p,9] = calcAcc(indxs_c,indxs_p)
          resultsErr[count_p,9] = calcErr(dists_c,dists_p)
          resultsRec[count_p,9] = calcRecall(indxs_c,indxs_p)
          resultsTime[count_p,9] = b_t
          resultsQps[count_p,9] = pointsQ/b_t
      end
      
      println("$(resultsTime[count_p,9]) secs")
      println("$(resultsAcc[count_p,9]) %")
      println("$(resultsRec[count_p,9]) recall")
      
      @everywhere CUDA.reclaim()
      @everywhere GC.gc(true)  
      
      println("\nCalculating Nearest Neighbours results")
      if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_nn_indxs,file_nn_dists,k,d, pointsC,pointsQ,C_size,Q_size,9,in_memory)
          
          resultsAcc[count_p,10] = calcAcc_bin(file_c_indxs,file_nn_indxs,k,Int32)
          resultsErr[count_p,10] = calcErr_bin(file_c_dists,file_nn_dists,k,Float32)
          resultsRec[count_p,10] = calcRecall_bin(file_c_indxs,file_nn_indxs,k,Int32)
          resultsTime[count_p,10] = b_t
          resultsQps[count_p,10] = pointsQ/b_t
      else
          indxs_nn = zeros(Int32,k,pointsQ)
          dists_nn = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_nn, dists_nn = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,9,in_memory)
          
          resultsAcc[count_p,10] = calcAcc(indxs_c,indxs_nn)
          resultsErr[count_p,10] = calcErr(dists_c,dists_nn)
          resultsRec[count_p,10] = calcRecall(indxs_c,indxs_nn)
          resultsTime[count_p,10] = b_t
          resultsQps[count_p,10] = pointsQ/b_t
      end
      
      println("$(resultsTime[count_p,10]) secs")
      println("$(resultsAcc[count_p,10]) %")
      println("$(resultsRec[count_p,10]) recall")

     @everywhere GC.gc(true)           
     
     println("\nCalculating Faiss results")
     if in_memory == false
          
          b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_fs_indxs,file_fs_dists,k,d, pointsC,pointsQ,C_size,Q_size,10,in_memory)
          
          resultsAcc[count_p,11] = calcAcc_bin(file_c_indxs,file_fs_indxs,k,Int32)
          resultsErr[count_p,11] = calcErr_bin(file_c_dists,file_fs_dists,k,Float32)
          resultsRec[count_p,11] = calcRecall_bin(file_c_indxs,file_fs_indxs,k,Int32)
          resultsTime[count_p,11] = b_t
          resultsQps[count_p,11] = pointsQ/b_t
      else
          indxs_fs = zeros(Int32,k,pointsQ)
          dists_fs = zeros(Float32,k,pointsQ)
          
          b_t = @elapsed indxs_fs, dists_fs = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size,Q_size,10,in_memory)
          
          resultsAcc[count_p,11] = calcAcc(indxs_c,indxs_fs)
          resultsErr[count_p,11] = calcErr(dists_c,dists_fs)
          resultsRec[count_p,11] = calcRecall(indxs_c,indxs_fs)
          resultsTime[count_p,11] = b_t
          resultsQps[count_p,11] = pointsQ/b_t
      end
      
     println("$(resultsTime[count_p,11]) secs")
     println("$(resultsAcc[count_p,11]) %")
     println("$(resultsRec[count_p,11]) recall")
     
     println("Comparing FLANN with GPU results")
     if in_memory == false
          resultsBest[count_p,1], resultsBest[count_p,2], resultsBest[count_p,3] = calcBest_bin(file_d_dists,file_g_dists,k,Float32)
     else
          resultsBest[count_p,1], resultsBest[count_p,2], resultsBest[count_p,3] = calcBest(dists_d,dists_g)
     end
     println("Algorithm comparisons results (best): $(resultsBest[count_p,1])% FLANN, $(resultsBest[count_p,2])% GPU, $(resultsBest[count_p,3])% Tie")
     
     println("Comparing FLANN with CPU results")
     if in_memory == false
          resultsBest2[count_p,1], resultsBest2[count_p,2], resultsBest2[count_p,3] = calcBest_bin(file_d_dists,file_c_dists,k,Float32)
     else
          resultsBest2[count_p,1], resultsBest2[count_p,2], resultsBest2[count_p,3] = calcBest(dists_d,dists_c)
     end
     println("Algorithm comparisons results (best): $(resultsBest2[count_p,1])% FLANN, $(resultsBest2[count_p,2])% CPU, $(resultsBest2[count_p,3])% Tie")
     
     println("Comparing CPU with GPU results")
     if in_memory == false
          resultsBest3[count_p,1], resultsBest3[count_p,2], resultsBest3[count_p,3] = calcBest_bin(file_c_dists,file_g_dists,k,Float32)
     else
          resultsBest3[count_p,1], resultsBest3[count_p,2], resultsBest3[count_p,3] = calcBest(dists_c,dists_g)
     end
     println("Algorithm comparisons results (best): $(resultsBest3[count_p,1])% CPU, $(resultsBest3[count_p,2])% GPU, $(resultsBest3[count_p,3])% Tie")
     
     println("Comparing GPU with ParallelKNN results")
     if in_memory == false
          resultsBest4[count_p,1], resultsBest4[count_p,2], resultsBest4[count_p,3] = calcBest_bin(file_g_dists,file_p_dists,k,Float32)
     else
          resultsBest4[count_p,1], resultsBest4[count_p,2], resultsBest4[count_p,3] = calcBest(dists_g,dists_p)
     end
     println("Algorithm comparisons results (best): $(resultsBest3[count_p,1])% CPU, $(resultsBest3[count_p,2])% GPU, $(resultsBest3[count_p,3])% Tie")
     
     @everywhere CUDA.reclaim()
     @everywhere GC.gc(true)
      
      indxs_d = nothing
      indxs_a = nothing
      indxs_g = nothing
      indxs_c = nothing
      indxs_p = nothing
      indxs_cc = nothing
      indxs_cg = nothing
      indxs_rpg = nothing
      indxs_rpc = nothing
      dists_d = nothing
      dists_a = nothing
      dists_g = nothing
      dists_c = nothing
      dists_p = nothing
      dists_cc = nothing
      dists_cg = nothing
      dists_rpg = nothing
      dists_rpc = nothing
      
      b_t = nothing
      
      @everywhere GC.gc(true)
      
      return nothing
end


function batchTest(
      filename_input_C::String,
      filename_input_Q::String,
      file_d_indxs::String,
      file_d_dists::String,
      file_a_indxs::String,
      file_a_dists::String,
      file_g_indxs::String,
      file_g_dists::String,
      file_c_indxs::String,
      file_c_dists::String,
      file_p_indxs::String,
      file_p_dists::String,
      file_cc_indxs::String,
      file_cc_dists::String,
      file_cg_indxs::String,
      file_cg_dists::String,
      file_rpc_indxs::String,
      file_rpc_dists::String,
      file_rpg_indxs::String,
      file_rpg_dists::String,
      file_nn_indxs::String,
      file_nn_dists::String,
      file_fs_indxs::String,
      file_fs_dists::String,
      k::Int = 150, 
      d::Int = 28,
      r::Int = 3,
      P::Int = 5,
      C_size:: Int = 10000,
      Q_size::Int = 5000,
      bench::Bool = false,
      in_memory::Bool = false
      )

      nPoints = 4
      nAlg = 11
      
      resultsAcc = zeros(Float64, nPoints,nAlg)
      resultsErr = zeros(Float64, nPoints,nAlg)
      resultsTime = zeros(Float64, nPoints,nAlg)
      resultsRec = zeros(Float64, nPoints,nAlg)
      resultsQps = zeros(Float64, nPoints,nAlg)
      
      resultsBest = zeros(Float64, nPoints,nAlg)
      resultsBest2 = zeros(Float64, nPoints,nAlg)
      resultsBest3 = zeros(Float64, nPoints,nAlg)
      resultsBest4 = zeros(Float64, nPoints,nAlg)
      
      pointsC = [10000 20000 40000 80000]
      if filename_input_C == filename_input_Q
         pointsQ = [10000 20000 40000 80000]
      else
         pointsQ = [2000 4000 8000 16000]
      end
      
      nam = ["10000" "20000" "40000" "60000"]
      alg = ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"]
      pal = palette(:Paired_11)
      
      
      cid = 1
      for count_p in 1:nPoints     
          println("=========================================")
          println("\t Starting loop for $(pointsC[count_p]) Corpus points and $(pointsQ[count_p]) Query points")
          println("=========================================")
          
          if C_size > pointsC[count_p]
             C_size = pointsC[count_p]
          end
      
          if Q_size > pointsQ[count_p]
             Q_size = pointsQ[count_p]
          end
      
          if k > pointsQ[count_p]
             k = pointsQ[count_p]
          end
          
          println("\nCalculating cpu results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_c_indxs,$file_c_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,3,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_c_indxs,file_c_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,3,in_memory)
              end
              resultsAcc[count_p,4] = 100.0
              resultsErr[count_p,4] = 0.0
              resultsRec[count_p,4] = 1.0
              resultsTime[count_p,4] = b_t
              resultsQps[count_p,4] = pointsQ[count_p]/b_t
          else
              indxs_c = zeros(Int32,k,pointsQ[count_p])
              dists_c = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_c, $dists_c = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,3,$in_memory)
              else
                  b_t = @elapsed indxs_c, dists_c = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,3,in_memory)
              end
              resultsAcc[count_p,4] = 100.0
              resultsErr[count_p,4] = 0.0
              resultsRec[count_p,4] = 1.0
              resultsTime[count_p,4] = b_t
              resultsQps[count_p,4] = pointsQ[count_p]/b_t
          end
          
         println("$(resultsTime[count_p,4]) secs")
         println("$(resultsAcc[count_p,4]) %")
         println("$(resultsRec[count_p,4]) recall")
         
         @everywhere GC.gc(true)
         
          println("\nCalculating FLANN brute force results")
          if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_d_indxs,$file_d_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,0,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_d_indxs,file_d_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,0,in_memory)
              end
              resultsAcc[count_p,1] = calcAcc_bin(file_c_indxs,file_d_indxs,k,Int32)
              resultsErr[count_p,1] = calcErr_bin(file_c_dists,file_d_dists,k,Float32)
              resultsRec[count_p,1] = calcRecall_bin(file_c_indxs,file_d_indxs,k,Int32)
              resultsTime[count_p,1] = b_t
              resultsQps[count_p,1] = pointsQ[count_p]/b_t
          else
              indxs_d = zeros(Int32,k,pointsQ[count_p])
              dists_d = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_d, $dists_d = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,0,$in_memory)
              else
                  b_t = @elapsed indxs_d, dists_d = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,0,in_memory)
              end
              resultsAcc[count_p,1] = calcAcc(indxs_c,indxs_d)
              resultsErr[count_p,1] = calcErr(dists_c,dists_d)
              resultsRec[count_p,1] = calcRecall(indxs_c,indxs_d)
              resultsTime[count_p,1] = b_t
              resultsQps[count_p,1] = pointsQ[count_p]/b_t
          end
          

          println("$(resultsTime[count_p,1]) secs")
          println("$(resultsAcc[count_p,1]) %")
          println("$(resultsRec[count_p,1]) recall")
      
      
         @everywhere GC.gc(true)
      
         println("\nCalculating FLANN approximated results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_a_indxs,$file_a_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,1,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_a_indxs,file_a_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,1,in_memory)
              end
              resultsAcc[count_p,2] = calcAcc_bin(file_c_indxs,file_a_indxs,k,Int32)
              resultsErr[count_p,2] = calcErr_bin(file_c_dists,file_a_dists,k,Float32)
              resultsRec[count_p,2] = calcRecall_bin(file_c_indxs,file_a_indxs,k,Int32)
              resultsTime[count_p,2] = b_t
              resultsQps[count_p,2] = pointsQ[count_p]/b_t
          else
              indxs_a = zeros(Int32,k,pointsQ[count_p])
              dists_a = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_a, $dists_a = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,1,$in_memory)
              else
                  b_t = @elapsed indxs_a, dists_a = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,1,in_memory)
              end
              resultsAcc[count_p,2] = calcAcc(indxs_c,indxs_a)
              resultsErr[count_p,2] = calcErr(dists_c,dists_a)
              resultsRec[count_p,2] = calcRecall(indxs_c,indxs_a)
              resultsTime[count_p,2] = b_t
              resultsQps[count_p,2] = pointsQ[count_p]/b_t
          end
          
         println("$(resultsTime[count_p,2]) secs")
         println("$(resultsAcc[count_p,2]) %")
         println("$(resultsRec[count_p,2]) recall")
      
      
         @everywhere GC.gc(true)
      
         println("\nCalculating brute force gpu results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_g_indxs,$file_g_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,2,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_g_indxs,file_g_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,2,in_memory)
              end
              resultsAcc[count_p,3] = calcAcc_bin(file_c_indxs,file_g_indxs,k,Int32)
              resultsErr[count_p,3] = calcErr_bin(file_c_dists,file_g_dists,k,Float32)
              resultsRec[count_p,3] = calcRecall_bin(file_c_indxs,file_g_indxs,k,Int32)
              resultsTime[count_p,3] = b_t
              resultsQps[count_p,3] = pointsQ[count_p]/b_t
          else
              indxs_g = zeros(Int32,k,pointsQ[count_p])
              dists_g = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_g, $dists_g = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,2,$in_memory)
              else
                  b_t = @elapsed indxs_g, dists_g = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,2,in_memory)
              end
              resultsAcc[count_p,3] = calcAcc(indxs_c,indxs_g)
              resultsErr[count_p,3] = calcErr(dists_c,dists_g)
              resultsRec[count_p,3] = calcRecall(indxs_c,indxs_g)
              resultsTime[count_p,3] = b_t
              resultsQps[count_p,3] = pointsQ[count_p]/b_t
          end
          
          println("$(resultsTime[count_p,3]) secs")
          println("$(resultsAcc[count_p,3]) %")
          println("$(resultsRec[count_p,3]) recall")
          
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)
         
          println("\nCalculating Random Projection GPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_rpg_indxs,$file_rpg_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,6,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_rpg_indxs,file_rpg_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,6,in_memory)
              end
              resultsAcc[count_p,5] = calcAcc_bin(file_c_indxs,file_rpg_indxs,k,Int32)
              resultsErr[count_p,5] = calcErr_bin(file_c_dists,file_rpg_dists,k,Float32)
              resultsRec[count_p,5] = calcRecall_bin(file_c_indxs,file_rpg_indxs,k,Int32)
              resultsTime[count_p,5] = b_t
              resultsQps[count_p,5] = pointsQ[count_p]/b_t
          else
              indxs_rpg = zeros(Int32,k,pointsQ[count_p])
              dists_rpg = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_rpg, $dists_rpg = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,6,$in_memory,$r,$P)
              else
                  b_t = @elapsed indxs_rpg, dists_rpg = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,6,in_memory,r,P)
              end
              resultsAcc[count_p,5] = calcAcc(indxs_c,indxs_rpg)
              resultsErr[count_p,5] = calcErr(dists_c,dists_rpg)
              resultsRec[count_p,5] = calcRecall(indxs_c,indxs_rpg)
              resultsTime[count_p,5] = b_t
              resultsQps[count_p,5] = pointsQ[count_p]/b_t
          end
          
          println("$(resultsTime[count_p,5]) secs")
          println("$(resultsAcc[count_p,5]) %")
          println("$(resultsRec[count_p,5]) recall")
          
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)
         
         
         println("\nCalculating Random Projection CPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_rpc_indxs,$file_rpc_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,5,$in_memory,$r,$P)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_rpc_indxs,file_rpc_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,5,in_memory,r,P)
              end
              resultsAcc[count_p,6] = calcAcc_bin(file_c_indxs,file_rpc_indxs,k,Int32)
              resultsErr[count_p,6] = calcErr_bin(file_c_dists,file_rpc_dists,k,Float32)
              resultsRec[count_p,6] = calcRecall_bin(file_c_indxs,file_rpc_indxs,k,Int32)
              resultsTime[count_p,6] = b_t
              resultsQps[count_p,6] = pointsQ[count_p]/b_t
          else
              indxs_rpc = zeros(Int32,k,pointsQ[count_p])
              dists_rpc = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_rpc, $dists_rpc = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,5,$in_memory,$r,$P)
              else
                  b_t = @elapsed indxs_rpc, dists_rpc = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,5,in_memory,r,P)
              end
              resultsAcc[count_p,6] = calcAcc(indxs_c,indxs_rpc)
              resultsErr[count_p,6] = calcErr(dists_c,dists_rpc)
              resultsRec[count_p,6] = calcRecall(indxs_c,indxs_rpc)
              resultsTime[count_p,6] = b_t
              resultsQps[count_p,6] = pointsQ[count_p]/b_t
          end
          
          println("$(resultsTime[count_p,6]) secs")
          println("$(resultsAcc[count_p,6]) %")
          println("$(resultsRec[count_p,6]) recall")
          
          
         println("\nCalculating Cluster TI GPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_cg_indxs,$file_cg_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,8,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_cg_indxs,file_cg_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,8,in_memory)
              end
              resultsAcc[count_p,7] = calcAcc_bin(file_c_indxs,file_cg_indxs,k,Int32)
              resultsErr[count_p,7] = calcErr_bin(file_c_dists,file_cg_dists,k,Float32)
              resultsRec[count_p,7] = calcRecall_bin(file_c_indxs,file_cg_indxs,k,Int32)
              resultsTime[count_p,7] = b_t
              resultsQps[count_p,7] = pointsQ[count_p]/b_t
          else
              indxs_cg = zeros(Int32,k,pointsQ[count_p])
              dists_cg = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_cg, $dists_cg = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,8,$in_memory)
              else
                  b_t = @elapsed indxs_cg, dists_cg = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,8,in_memory)
              end
              resultsAcc[count_p,7] = calcAcc(indxs_c,indxs_cg)
              resultsErr[count_p,7] = calcErr(dists_c,dists_cg)
              resultsRec[count_p,7] = calcRecall(indxs_c,indxs_cg)
              resultsTime[count_p,7] = b_t
              resultsQps[count_p,7] = pointsQ[count_p]/b_t
          end
          
          println("$(resultsTime[count_p,7]) secs")
          println("$(resultsAcc[count_p,7]) %")
          println("$(resultsRec[count_p,7]) recall")
         
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)          
          
          
         println("\nCalculating Cluster TI CPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_cc_indxs,$file_cc_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,7,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_cc_indxs,file_cc_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,7,in_memory)
              end
              resultsAcc[count_p,8] = calcAcc_bin(file_c_indxs,file_cc_indxs,k,Int32)
              resultsErr[count_p,8] = calcErr_bin(file_c_dists,file_cc_dists,k,Float32)
              resultsRec[count_p,8] = calcRecall_bin(file_c_indxs,file_cc_indxs,k,Int32)
              resultsTime[count_p,8] = b_t
              resultsQps[count_p,8] = pointsQ[count_p]/b_t
          else
              indxs_cc = zeros(Int32,k,pointsQ[count_p])
              dists_cc = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_cc, $dists_cc = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,7,$in_memory)
              else
                  b_t = @elapsed indxs_cc, dists_cc = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,7,in_memory)
              end
              resultsAcc[count_p,8] = calcAcc(indxs_c,indxs_cc)
              resultsErr[count_p,8] = calcErr(dists_c,dists_cc)
              resultsRec[count_p,8] = calcRecall(indxs_c,indxs_cc)
              resultsTime[count_p,8] = b_t
              resultsQps[count_p,8] = pointsQ[count_p]/b_t
          end
          
          println("$(resultsTime[count_p,8]) secs")
          println("$(resultsAcc[count_p,8]) %")
          println("$(resultsRec[count_p,8]) recall")

         @everywhere GC.gc(true)           
         
         println("\nCalculating ParallelKNN (gpu) results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_p_indxs,$file_p_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,4,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_p_indxs,file_p_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,4,in_memory)
              end
              resultsAcc[count_p,9] = calcAcc_bin(file_c_indxs,file_p_indxs,k,Int32)
              resultsErr[count_p,9] = calcErr_bin(file_c_dists,file_p_dists,k,Float32)
              resultsRec[count_p,9] = calcRecall_bin(file_c_indxs,file_p_indxs,k,Int32)
              resultsTime[count_p,9] = b_t
              resultsQps[count_p,9] = pointsQ[count_p]/b_t
          else
              indxs_p = zeros(Int32,k,pointsQ[count_p])
              dists_p = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_p, $dists_p = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,4,$in_memory)
              else
                  b_t = @elapsed indxs_p, dists_p = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,4,in_memory)
              end
              resultsAcc[count_p,9] = calcAcc(indxs_c,indxs_p)
              resultsErr[count_p,9] = calcErr(dists_c,dists_p)
              resultsRec[count_p,9] = calcRecall(indxs_c,indxs_p)
              resultsTime[count_p,9] = b_t
              resultsQps[count_p,9] = pointsQ[count_p]/b_t
          end
          
          println("$(resultsTime[count_p,9]) secs")
          println("$(resultsAcc[count_p,9]) %")
          println("$(resultsRec[count_p,9]) recall")
          
          @everywhere CUDA.reclaim()
          @everywhere GC.gc(true)
          
         println("\nCalculating Nearest Neighbours results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_nn_indxs,$file_nn_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,9,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_nn_indxs,file_nn_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,9,in_memory)
              end
              resultsAcc[count_p,10] = calcAcc_bin(file_c_indxs,file_nn_indxs,k,Int32)
              resultsErr[count_p,10] = calcErr_bin(file_c_dists,file_nn_dists,k,Float32)
              resultsRec[count_p,10] = calcRecall_bin(file_c_indxs,file_nn_indxs,k,Int32)
              resultsTime[count_p,10] = b_t
              resultsQps[count_p,10] = pointsQ[count_p]/b_t
          else
              indxs_nn = zeros(Int32,k,pointsQ[count_p])
              dists_nn = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_nn, $dists_nn = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,9,$in_memory)
              else
                  b_t = @elapsed indxs_nn, dists_nn = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,9,in_memory)
              end
              resultsAcc[count_p,10] = calcAcc(indxs_c,indxs_nn)
              resultsErr[count_p,10] = calcErr(dists_c,dists_nn)
              resultsRec[count_p,10] = calcRecall(indxs_c,indxs_nn)
              resultsTime[count_p,10] = b_t
              resultsQps[count_p,10] = pointsQ[count_p]/b_t
          end
          
          println("$(resultsTime[count_p,10]) secs")
          println("$(resultsAcc[count_p,10]) %")
          println("$(resultsRec[count_p,10]) recall")

         @everywhere GC.gc(true)           
         
         println("\nCalculating Faiss results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_fs_indxs,$file_fs_dists,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,10,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_fs_indxs,file_fs_dists,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,10,in_memory)
              end
              resultsAcc[count_p,11] = calcAcc_bin(file_c_indxs,file_fs_indxs,k,Int32)
              resultsErr[count_p,11] = calcErr_bin(file_c_dists,file_fs_dists,k,Float32)
              resultsRec[count_p,11] = calcRecall_bin(file_c_indxs,file_fs_indxs,k,Int32)
              resultsTime[count_p,11] = b_t
              resultsQps[count_p,11] = pointsQ[count_p]/b_t
          else
              indxs_fs = zeros(Int32,k,pointsQ[count_p])
              dists_fs = zeros(Float32,k,pointsQ[count_p])
              if bench == true
                  b_t = @belapsed $indxs_fs, $dists_fs = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC[$count_p],$pointsQ[$count_p],$C_size,$Q_size,10,$in_memory)
              else
                  b_t = @elapsed indxs_fs, dists_fs = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC[count_p],pointsQ[count_p],C_size,Q_size,10,in_memory)
              end
              resultsAcc[count_p,11] = calcAcc(indxs_c,indxs_fs)
              resultsErr[count_p,11] = calcErr(dists_c,dists_fs)
              resultsRec[count_p,11] = calcRecall(indxs_c,indxs_fs)
              resultsTime[count_p,11] = b_t
              resultsQps[count_p,11] = pointsQ[count_p]/b_t
          end
          
         println("$(resultsTime[count_p,11]) secs")
         println("$(resultsAcc[count_p,11]) %")
         println("$(resultsRec[count_p,11]) recall")
         
         println("Comparing FLANN with GPU results")
         if in_memory == false
              resultsBest[count_p,1], resultsBest[count_p,2], resultsBest[count_p,3] = calcBest_bin(file_d_dists,file_g_dists,k,Float32)
         else
              resultsBest[count_p,1], resultsBest[count_p,2], resultsBest[count_p,3] = calcBest(dists_d,dists_g)
         end
         println("Algorithm comparisons results (best): $(resultsBest[count_p,1])% FLANN, $(resultsBest[count_p,2])% GPU, $(resultsBest[count_p,3])% Tie")
         
         println("Comparing FLANN with CPU results")
         if in_memory == false
              resultsBest2[count_p,1], resultsBest2[count_p,2], resultsBest2[count_p,3] = calcBest_bin(file_d_dists,file_c_dists,k,Float32)
         else
              resultsBest2[count_p,1], resultsBest2[count_p,2], resultsBest2[count_p,3] = calcBest(dists_d,dists_c)
         end
         println("Algorithm comparisons results (best): $(resultsBest2[count_p,1])% FLANN, $(resultsBest2[count_p,2])% CPU, $(resultsBest2[count_p,3])% Tie")
         
         println("Comparing CPU with GPU results")
         if in_memory == false
              resultsBest3[count_p,1], resultsBest3[count_p,2], resultsBest3[count_p,3] = calcBest_bin(file_c_dists,file_g_dists,k,Float32)
         else
              resultsBest3[count_p,1], resultsBest3[count_p,2], resultsBest3[count_p,3] = calcBest(dists_c,dists_g)
         end
         println("Algorithm comparisons results (best): $(resultsBest3[count_p,1])% CPU, $(resultsBest3[count_p,2])% GPU, $(resultsBest3[count_p,3])% Tie")
         
         println("Comparing GPU with ParallelKNN results")
         if in_memory == false
              resultsBest4[count_p,1], resultsBest4[count_p,2], resultsBest4[count_p,3] = calcBest_bin(file_g_dists,file_p_dists,k,Float32)
         else
              resultsBest4[count_p,1], resultsBest4[count_p,2], resultsBest4[count_p,3] = calcBest(dists_g,dists_p)
         end
         println("Algorithm comparisons results (best): $(resultsBest3[count_p,1])% CPU, $(resultsBest3[count_p,2])% GPU, $(resultsBest3[count_p,3])% Tie")
         
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)
         
         
      end
      
       
      println("Generating result graphs")
      
      p = plot(nam[1,:], resultsTime, title = "Performance comparison", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Time (sec)", xlabel = "Points (number)",size = (1600, 1400), palette = pal);
      
      p2 = plot(nam[1,:], resultsAcc, title = "Accuracy comparison", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Accuracy (%)", xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p3 = plot(nam[1,:], resultsErr, title = "Error comparison", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Error (std)",xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p4 = plot(nam[1,:], resultsBest, title = "Algorithm results comparison FLANN - GPU", label= ["FLANN (bf) wins" "GPU wins" "Tie"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Percent of wins for each algorithm (%)",xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p5 = plot(nam[1,:], resultsBest2, title = "Algorithm results comparison FLANN - CPU", label= ["FLANN (bf) wins" "CPU wins" "Tie"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Percent of wins for each algorithm (%)",xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p6 = plot(nam[1,:], resultsBest3, title = "Algorithm results comparison CPU - GPU", label= ["CPU wins" "GPU wins" "Tie"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Percent of wins for each algorithm (%)",xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p7 = plot(nam[1,:], resultsBest4, title = "Algorithm results comparison GPU - ParallelKNN", label= ["GPU wins" "ParallelKNN wins" "Tie"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Percent of wins for each algorithm (%)",xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p8 = plot(nam[1,:], resultsRec, title = "Recall comparison", label= ["FLANN (bf)" "FLANN (app)" "GPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Recall", xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p9 = plot(resultsRec, resultsQps, title = "Recall-Queries per second", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Queries per second", xlabel = "Recall",size = (1600, 1400), palette = pal);
      
      
      savefig(p, "./Graphs/Time_comparisons.png")
      
      savefig(p2, "./Graphs/Accuracy_comparisons.png")
      
      savefig(p3, "./Graphs/Error_comparisons.png")
      
      savefig(p4, "./Graphs/Algorithm_comparisons_(FLANN-GPU).png")
      
      savefig(p5, "./Graphs/Algorithm_comparisons_(FLANN-CPU).png")
      
      savefig(p6, "./Graphs/Algorithm_comparisons_(CPU-GPU).png")
      
      savefig(p7, "./Graphs/Algorithm_comparisons_(GPU-ParallelKNN).png")
      
      savefig(p8, "./Graphs/Recall_comparisons.png")
      
      savefig(p9, "./Graphs/QPSRecall_comparisons.png")
      
      indxs_d = nothing
      indxs_a = nothing
      indxs_g = nothing
      indxs_c = nothing
      indxs_p = nothing
      indxs_cc = nothing
      indxs_cg = nothing
      indxs_rpg = nothing
      indxs_rpc = nothing
      indxs_nn = nothing
      indx_fs = nothing
      dists_d = nothing
      dists_a = nothing
      dists_g = nothing
      dists_c = nothing
      dists_p = nothing
      dists_cc = nothing
      dists_cg = nothing
      dists_rpg = nothing
      dists_rpc = nothing
      dists_nn = nothing
      dists_fs = nothing
      
      p = nothing
      p2 = nothing
      p3 = nothing
      p4 = nothing
      p5 = nothing
      p6 = nothing
      p7 = nothing
      p8 = nothing
      p9 = nothing
      
      b_t = nothing
      
      @everywhere GC.gc(true)
      
      return nothing
end

function batchTest2(
      filename_input_C::String,
      filename_input_Q::String,
      file_d_indxs::String,
      file_d_dists::String,
      file_a_indxs::String,
      file_a_dists::String,
      file_g_indxs::String,
      file_g_dists::String,
      file_c_indxs::String,
      file_c_dists::String,
      file_p_indxs::String,
      file_p_dists::String,
      file_cc_indxs::String,
      file_cc_dists::String,
      file_cg_indxs::String,
      file_cg_dists::String,
      file_rpc_indxs::String,
      file_rpc_dists::String,
      file_rpg_indxs::String,
      file_rpg_dists::String,
      file_nn_indxs::String,
      file_nn_dists::String,
      file_fs_indxs::String,
      file_fs_dists::String,
      k::Int = 150, 
      d::Int = 28,
      r::Int = 3,
      P::Int = 5,
      bench::Bool = false,
      in_memory::Bool = false
      )

      nPoints = 4
      nAlg = 11
      
      resultsAcc = zeros(Float64, nPoints,nAlg)
      resultsErr = zeros(Float64, nPoints,nAlg)
      resultsRec = zeros(Float64, nPoints,nAlg)
      resultsTime = zeros(Float64, nPoints,nAlg)
      resultsQps = zeros(Float64, nPoints,nAlg)
      
      pointsC = 80000
      if filename_input_C == filename_input_Q
         pointsQ = 80000
      else
         pointsQ = 16000
      end
      
      C_size = [1600 3200 6400 12800]
      Q_size = [1600 3200 6400 12800]
      
      nam = ["1600" "3200" "6400" "12800"]
      alg = ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"]
      pal = palette(:Paired_11)
      
      
      cid = 1
      for count_p in 1:nPoints     
          println("=========================================")
          println("\t Starting loop for $(pointsC) Corpus points and $(pointsQ) Query points with cSize $(C_size[count_p]) and qSize $(Q_size[count_p])")
          println("=========================================")
          
          if C_size[count_p] > pointsC
             C_size[count_p] = pointsC
          end
      
          if Q_size[count_p] > pointsQ
             Q_size[count_p] = pointsQ
          end
      
          if k > pointsQ
             k = pointsQ
          end
          
          println("\nCalculating cpu results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_c_indxs,$file_c_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],3,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_c_indxs,file_c_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],3,in_memory)
              end
              resultsAcc[count_p,4] = 100.0
              resultsErr[count_p,4] = 0.0
              resultsRec[count_p,4] = 1.0
              resultsTime[count_p,4] = b_t
              resultsQps[count_p,4] = pointsQ/b_t
          else
              indxs_c = zeros(Int32,k,pointsQ)
              dists_c = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_c, $dists_c = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],3,$in_memory)
              else
                  b_t = @elapsed indxs_c, dists_c = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],3,in_memory)
              end
              resultsAcc[count_p,4] = 100.0
              resultsErr[count_p,4] = 0.0
              resultsRec[count_p,4] = 1.0
              resultsTime[count_p,4] = b_t
              resultsQps[count_p,4] = pointsQ/b_t
          end
          
         println("$(resultsTime[count_p,4]) secs")
         println("$(resultsAcc[count_p,4]) %")
         println("$(resultsRec[count_p,4]) recall")
         
         @everywhere GC.gc(true)
         
          println("\nCalculating FLANN brute force results")
          if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_d_indxs,$file_d_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],0,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_d_indxs,file_d_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],0,in_memory)
              end
              resultsAcc[count_p,1] = calcAcc_bin(file_c_indxs,file_d_indxs,k,Int32)
              resultsErr[count_p,1] = calcErr_bin(file_c_dists,file_d_dists,k,Float32)
              resultsRec[count_p,1] = calcRecall_bin(file_c_indxs,file_d_indxs,k,Int32)
              resultsTime[count_p,1] = b_t
              resultsQps[count_p,1] = pointsQ/b_t
          else
              indxs_d = zeros(Int32,k,pointsQ)
              dists_d = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_d, $dists_d = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],0,$in_memory)
              else
                  b_t = @elapsed indxs_d, dists_d = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],0,in_memory)
              end
              resultsAcc[count_p,1] = calcAcc(indxs_c,indxs_d)
              resultsErr[count_p,1] = calcErr(dists_c,dists_d)
              resultsRec[count_p,1] = calcRecall(indxs_c,indxs_d)
              resultsTime[count_p,1] = b_t
              resultsQps[count_p,1] = pointsQ/b_t
          end
          

          println("$(resultsTime[count_p,1]) secs")
          println("$(resultsAcc[count_p,1]) %")
          println("$(resultsRec[count_p,1]) recall")
      
      
         @everywhere GC.gc(true)
      
         println("\nCalculating FLANN approximated results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_a_indxs,$file_a_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],1,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_a_indxs,file_a_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],1,in_memory)
              end
              resultsAcc[count_p,2] = calcAcc_bin(file_c_indxs,file_a_indxs,k,Int32)
              resultsErr[count_p,2] = calcErr_bin(file_c_dists,file_a_dists,k,Float32)
              resultsRec[count_p,2] = calcRecall_bin(file_c_indxs,file_a_indxs,k,Int32)
              resultsTime[count_p,2] = b_t
              resultsQps[count_p,2] = pointsQ/b_t
          else
              indxs_a = zeros(Int32,k,pointsQ)
              dists_a = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_a, $dists_a = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],1,$in_memory)
              else
                  b_t = @elapsed indxs_a, dists_a = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],1,in_memory)
              end
              resultsAcc[count_p,2] = calcAcc(indxs_c,indxs_a)
              resultsErr[count_p,2] = calcErr(dists_c,dists_a)
              resultsRec[count_p,2] = calcRecall(indxs_c,indxs_a)
              resultsTime[count_p,2] = b_t
              resultsQps[count_p,2] = pointsQ/b_t
          end
          
         println("$(resultsTime[count_p,2]) secs")
         println("$(resultsAcc[count_p,2]) %")
         println("$(resultsRec[count_p,2]) recall")
      
      
         @everywhere GC.gc(true)
      
         println("\nCalculating brute force gpu results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_g_indxs,$file_g_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],2,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_g_indxs,file_g_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],2,in_memory)
              end
              resultsAcc[count_p,3] = calcAcc_bin(file_c_indxs,file_g_indxs,k,Int32)
              resultsErr[count_p,3] = calcErr_bin(file_c_dists,file_g_dists,k,Float32)
              resultsRec[count_p,3] = calcRecall_bin(file_c_indxs,file_g_indxs,k,Int32)
              resultsTime[count_p,3] = b_t
              resultsQps[count_p,3] = pointsQ/b_t
          else
              indxs_g = zeros(Int32,k,pointsQ)
              dists_g = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_g, $dists_g = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],2,$in_memory)
              else
                  b_t = @elapsed indxs_g, dists_g = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],2,in_memory)
              end
              resultsAcc[count_p,3] = calcAcc(indxs_c,indxs_g)
              resultsErr[count_p,3] = calcErr(dists_c,dists_g)
              resultsRec[count_p,3] = calcRecall(indxs_c,indxs_g)
              resultsTime[count_p,3] = b_t
              resultsQps[count_p,3] = pointsQ/b_t
          end
          
          println("$(resultsTime[count_p,3]) secs")
          println("$(resultsAcc[count_p,3]) %")
          println("$(resultsRec[count_p,3]) recall")
          
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)
         
          println("\nCalculating Random Projection GPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_rpg_indxs,$file_rpg_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],6,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_rpg_indxs,file_rpg_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],6,in_memory)
              end
              resultsAcc[count_p,5] = calcAcc_bin(file_c_indxs,file_rpg_indxs,k,Int32)
              resultsErr[count_p,5] = calcErr_bin(file_c_dists,file_rpg_dists,k,Float32)
              resultsRec[count_p,5] = calcRecall_bin(file_c_indxs,file_rpg_indxs,k,Int32)
              resultsTime[count_p,5] = b_t
              resultsQps[count_p,5] = pointsQ/b_t
          else
              indxs_rpg = zeros(Int32,k,pointsQ)
              dists_rpg = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_rpg, $dists_rpg = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],6,$in_memory,$r,$P)
              else
                  b_t = @elapsed indxs_rpg, dists_rpg = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],6,in_memory,r,P)
              end
              resultsAcc[count_p,5] = calcAcc(indxs_c,indxs_rpg)
              resultsErr[count_p,5] = calcErr(dists_c,dists_rpg)
              resultsRec[count_p,5] = calcRecall(indxs_c,indxs_rpg)
              resultsTime[count_p,5] = b_t
              resultsQps[count_p,5] = pointsQ/b_t
          end
          
          println("$(resultsTime[count_p,5]) secs")
          println("$(resultsAcc[count_p,5]) %")
          println("$(resultsRec[count_p,5]) recall")
          
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)
         
         
         println("\nCalculating Random Projection CPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_rpc_indxs,$file_rpc_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],5,$in_memory,$r,$P)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_rpc_indxs,file_rpc_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],5,in_memory,r,P)
              end
              resultsAcc[count_p,6] = calcAcc_bin(file_c_indxs,file_rpc_indxs,k,Int32)
              resultsErr[count_p,6] = calcErr_bin(file_c_dists,file_rpc_dists,k,Float32)
              resultsRec[count_p,6] = calcRecall_bin(file_c_indxs,file_rpc_indxs,k,Int32)
              resultsTime[count_p,6] = b_t
              resultsQps[count_p,6] = pointsQ/b_t
          else
              indxs_rpc = zeros(Int32,k,pointsQ)
              dists_rpc = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_rpc, $dists_rpc = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],5,$in_memory,$r,$P)
              else
                  b_t = @elapsed indxs_rpc, dists_rpc = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],5,in_memory,r,P)
              end
              resultsAcc[count_p,6] = calcAcc(indxs_c,indxs_rpc)
              resultsErr[count_p,6] = calcErr(dists_c,dists_rpc)
              resultsRec[count_p,6] = calcRecall(indxs_c,indxs_rpc)
              resultsTime[count_p,6] = b_t
              resultsQps[count_p,6] = pointsQ/b_t
          end
          
          println("$(resultsTime[count_p,6]) secs")
          println("$(resultsAcc[count_p,6]) %")
          println("$(resultsRec[count_p,6]) recall")
          
          
         println("\nCalculating Cluster TI GPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_cg_indxs,$file_cg_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],8,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_cg_indxs,file_cg_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],8,in_memory)
              end
              resultsAcc[count_p,7] = calcAcc_bin(file_c_indxs,file_cg_indxs,k,Int32)
              resultsErr[count_p,7] = calcErr_bin(file_c_dists,file_cg_dists,k,Float32)
              resultsRec[count_p,7] = calcRecall_bin(file_c_indxs,file_cg_indxs,k,Int32)
              resultsTime[count_p,7] = b_t
              resultsQps[count_p,7] = pointsQ/b_t
          else
              indxs_cg = zeros(Int32,k,pointsQ)
              dists_cg = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_cg, $dists_cg = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],8,$in_memory)
              else
                  b_t = @elapsed indxs_cg, dists_cg = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],8,in_memory)
              end
              resultsAcc[count_p,7] = calcAcc(indxs_c,indxs_cg)
              resultsErr[count_p,7] = calcErr(dists_c,dists_cg)
              resultsRec[count_p,7] = calcRecall(indxs_c,indxs_cg)
              resultsTime[count_p,7] = b_t
              resultsQps[count_p,7] = pointsQ/b_t
          end
          
          println("$(resultsTime[count_p,7]) secs")
          println("$(resultsAcc[count_p,7]) %")
          println("$(resultsRec[count_p,7]) recall")
         
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)          
          
          
         println("\nCalculating Cluster TI CPU results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_cc_indxs,$file_cc_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],7,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_cc_indxs,file_cc_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],7,in_memory)
              end
              resultsAcc[count_p,8] = calcAcc_bin(file_c_indxs,file_cc_indxs,k,Int32)
              resultsErr[count_p,8] = calcErr_bin(file_c_dists,file_cc_dists,k,Float32)
              resultsRec[count_p,8] = calcRecall_bin(file_c_indxs,file_cc_indxs,k,Int32)
              resultsTime[count_p,8] = b_t
              resultsQps[count_p,8] = pointsQ/b_t
          else
              indxs_cc = zeros(Int32,k,pointsQ)
              dists_cc = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_cc, $dists_cc = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],7,$in_memory)
              else
                  b_t = @elapsed indxs_cc, dists_cc = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],7,in_memory)
              end
              resultsAcc[count_p,8] = calcAcc(indxs_c,indxs_cc)
              resultsErr[count_p,8] = calcErr(dists_c,dists_cc)
              resultsRec[count_p,8] = calcRecall(indxs_c,indxs_cc)
              resultsTime[count_p,8] = b_t
              resultsQps[count_p,8] = pointsQ/b_t
          end
          
          println("$(resultsTime[count_p,8]) secs")
          println("$(resultsAcc[count_p,8]) %")
          println("$(resultsRec[count_p,8]) recall")

         @everywhere GC.gc(true)           
         
         println("\nCalculating ParallelKNN (gpu) results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_p_indxs,$file_p_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],4,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_p_indxs,file_p_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],4,in_memory)
              end
              resultsAcc[count_p,9] = calcAcc_bin(file_c_indxs,file_p_indxs,k,Int32)
              resultsErr[count_p,9] = calcErr_bin(file_c_dists,file_p_dists,k,Float32)
              resultsRec[count_p,9] = calcRecall_bin(file_c_indxs,file_p_indxs,k,Int32)
              resultsTime[count_p,9] = b_t
              resultsQps[count_p,9] = pointsQ/b_t
          else
              indxs_p = zeros(Int32,k,pointsQ)
              dists_p = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_p, $dists_p = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],4,$in_memory)
              else
                  b_t = @elapsed indxs_p, dists_p = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],4,in_memory)
              end
              resultsAcc[count_p,9] = calcAcc(indxs_c,indxs_p)
              resultsErr[count_p,9] = calcErr(dists_c,dists_p)
              resultsRec[count_p,9] = calcRecall(indxs_c,indxs_p)
              resultsTime[count_p,9] = b_t
              resultsQps[count_p,9] = pointsQ/b_t
          end
          
         println("$(resultsTime[count_p,9]) secs")
         println("$(resultsAcc[count_p,9]) %")
         println("$(resultsRec[count_p,9]) recall")
         
         
         @everywhere CUDA.reclaim()
         @everywhere GC.gc(true)
         
         println("\nCalculating Nearest Neighbours results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_nn_indxs,$file_nn_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],9,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_nn_indxs,file_nn_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],9,in_memory)
              end
              resultsAcc[count_p,10] = calcAcc_bin(file_c_indxs,file_nn_indxs,k,Int32)
              resultsErr[count_p,10] = calcErr_bin(file_c_dists,file_nn_dists,k,Float32)
              resultsRec[count_p,10] = calcRecall_bin(file_c_indxs,file_nn_indxs,k,Int32)
              resultsTime[count_p,10] = b_t
              resultsQps[count_p,10] = pointsQ/b_t
          else
              indxs_nn = zeros(Int32,k,pointsQ)
              dists_nn = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_nn, $dists_nn = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],9,$in_memory)
              else
                  b_t = @elapsed indxs_nn, dists_nn = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],9,in_memory)
              end
              resultsAcc[count_p,10] = calcAcc(indxs_c,indxs_nn)
              resultsErr[count_p,10] = calcErr(dists_c,dists_nn)
              resultsRec[count_p,10] = calcRecall(indxs_c,indxs_nn)
              resultsTime[count_p,10] = b_t
              resultsQps[count_p,10] = pointsQ/b_t
          end
          
          println("$(resultsTime[count_p,10]) secs")
          println("$(resultsAcc[count_p,10]) %")
          println("$(resultsRec[count_p,10]) recall")

         @everywhere GC.gc(true)           
         
         println("\nCalculating Faiss results")
         if in_memory == false
              if bench == true
                  b_t = @belapsed distributedKNN($filename_input_C,$filename_input_Q,$file_fs_indxs,$file_fs_dists,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],10,$in_memory)
              else
                  b_t = @elapsed distributedKNN(filename_input_C,filename_input_Q,file_fs_indxs,file_fs_dists,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],10,in_memory)
              end
              resultsAcc[count_p,11] = calcAcc_bin(file_c_indxs,file_fs_indxs,k,Int32)
              resultsErr[count_p,11] = calcErr_bin(file_c_dists,file_fs_dists,k,Float32)
              resultsRec[count_p,11] = calcRecall_bin(file_c_indxs,file_fs_indxs,k,Int32)
              resultsTime[count_p,11] = b_t
              resultsQps[count_p,11] = pointsQ/b_t
          else
              indxs_fs = zeros(Int32,k,pointsQ)
              dists_fs = zeros(Float32,k,pointsQ)
              if bench == true
                  b_t = @belapsed $indxs_fs, $dists_fs = distributedKNN($filename_input_C,$filename_input_Q,$k,$d, $pointsC,$pointsQ,$C_size[$count_p],$Q_size[$count_p],10,$in_memory)
              else
                  b_t = @elapsed indxs_fs, dists_fs = distributedKNN(filename_input_C,filename_input_Q,k,d, pointsC,pointsQ,C_size[count_p],Q_size[count_p],10,in_memory)
              end
              resultsAcc[count_p,11] = calcAcc(indxs_c,indxs_fs)
              resultsErr[count_p,11] = calcErr(dists_c,dists_fs)
              resultsRec[count_p,11] = calcRecall(indxs_c,indxs_fs)
              resultsTime[count_p,11] = b_t
              resultsQps[count_p,11] = pointsQ/b_t
          end
          
         println("$(resultsTime[count_p,11]) secs")
         println("$(resultsAcc[count_p,11]) %")
         println("$(resultsRec[count_p,11]) recall")
         
         
      end
      
      
      resultsTimeNoNN = zeros(Float64, nPoints,nAlg-1)
      
      resultsTimeNoNN[:,1:9] .= resultsTime[:,1:9]
      resultsTimeNoNN[:,10] .= resultsTime[:,11]
      
      println("Generating result graphs")
      
      p = plot(nam[1,:], resultsTime, title = "Performance comparison for different particion sizes (at $pointsC x $pointsQ input)", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Time (sec)", xlabel = "qSize and cSize (points)",size = (1600, 1400), palette = pal);
      
      p2 = plot(nam[1,:], resultsAcc, title = "Accuracy comparison", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Accuracy (%)", xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p3 = plot(nam[1,:], resultsErr, title = "Error comparison", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Error (std)",xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p4 = plot(nam[1,:], resultsRec, title = "Recall comparison", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Recall", xlabel = "Points (number)", size = (1600, 1400), palette = pal);
      
      p5 = plot(resultsRec, resultsQps, title = "Recall-Queries per second (at $pointsC x $pointsQ input)", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "NearestNeighbours" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Queries per second", xlabel = "Recall",size = (1600, 1400), palette = pal);
      
      p6 = plot(nam[1,:], resultsTimeNoNN, title = "Performance comparison for different particion sizes (at $pointsC x $pointsQ input) (No NN)", label= ["FLANN (bf)" "FLANN (app)" "GPU" "CPU" "Rand Pr (GPU)" "Rand Pr (CPU)" "Cluster TI (GPU)" "Cluster TI (CPU)" "ParallelKNN" "Faiss"], left_margin = 20mm, bottom_margin = 20mm, lw = 4, ylabel = "Time (sec)", xlabel = "qSize and cSize (points)",size = (1600, 1400), palette = pal);
      
      savefig(p, "./Graphs/Partition_Time_comparisons.png")
      
      savefig(p2, "./Graphs/Partition_Accuracy_comparisons.png")
      
      savefig(p3, "./Graphs/Partition_Error_comparisons.png")
      
      savefig(p4, "./Graphs/Partition_Recall_comparisons.png")
      
      savefig(p5, "./Graphs/Partition_QPSRecall_comparisons.png")
      
      savefig(p6, "./Graphs/Partition_Time_comparisons_NoNN.png")
      
      indxs_d = nothing
      indxs_a = nothing
      indxs_g = nothing
      indxs_c = nothing
      indxs_p = nothing
      indxs_cc = nothing
      indxs_cg = nothing
      indxs_rpg = nothing
      indxs_rpc = nothing
      indxs_nn = nothing
      indxs_fs = nothing
      dists_d = nothing
      dists_a = nothing
      dists_g = nothing
      dists_c = nothing
      dists_p = nothing
      dists_cc = nothing
      dists_cg = nothing
      dists_rpg = nothing
      dists_rpc = nothing
      indxs_nn = nothing
      indxs_fs = nothing
      
      p = nothing
      p2 = nothing
      p3 = nothing
      p4 = nothing
      p5 = nothing
      p6 = nothing

      b_t = nothing
      
      @everywhere GC.gc(true)
      
      return nothing
end

batchWarmup(filename_input_C, filename_input_Q, 5, 2800, 3, 3, 3200, 3200, true)


batchTest(filename_input_C, filename_input_Q, file_d_indxs, file_d_dists, file_a_indxs,file_a_dists, file_g_indxs, file_g_dists, file_c_indxs, file_c_dists, file_p_indxs, file_p_dists, file_cc_indxs, file_cc_dists, file_cg_indxs, file_cg_dists, file_rpc_indxs, file_rpc_dists, file_rpg_indxs, file_rpg_dists, file_nn_indxs, file_nn_dists, file_fs_indxs, file_fs_dists, 150, 2800, 56, 14, 3200, 3200, false, true)


      
      

