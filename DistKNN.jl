using FLANN, BenchmarkTools, Statistics, Distributed, DelimitedFiles, ParallelNeighbors, CUDA

include("read_file.jl")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~                 Utilities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@inline function fast_sortperm(
         Distance::AbstractArray{Float32},
         k::Int
         )
         di = [Pair(i, d) for (i,d) in enumerate(Distance)]
         
         @inbounds sort!(di, by=x->x.second, alg=PartialQuickSort(1:k))

         return @inbounds [d.first for d in di[1:k]]

end

@inline function fast_sortperm_offset(
         Distance::AbstractArray{Float32},
         k::Int,
         offset::Int
         )
         di = [Pair(i, d) for (i,d) in enumerate(Distance)]
         
         @inbounds sort!(di, by=x->x.second, alg=PartialQuickSort(1:k))

         return @inbounds ([d.first + offset for d in di[1:k]], [d.second for d in di[1:k]])

end

@inline function fast_sortperm_offset(
         Distance::AbstractArray{Float32},
         Indxs_out::AbstractArray{Int32},
         Dists_out::AbstractArray{Float32},
         k::Int,
         offset::Int
         )
         di = [Pair(i, d) for (i,d) in enumerate(Distance)]
         
         if size(Indxs_out,1) < k || size(Dists_out,1) < k
             error("The input and output arrays for the fast_sortperm function have less than k dimensions")
         end
         if size(Indxs_out,2) < size(Distance,2) || size(Dists_out,2) < size(Distance,2)
             error("The input and output arrays for the fast_sortperm function have different sizes")
         end
         
         @inbounds sort!(di, by=x->x.second, alg=PartialQuickSort(1:k))
         
         i = 1
         @inbounds @views for d in di[1:k]
             Indxs_out[i] = d.first + offset
             Dists_out[i] = d.second
             i += 1
         end
         
         di = nothing
         return
end

@inline function fast_sortperm_offset(
         Distance::AbstractArray{Float32},
         Indxs_out::AbstractArray{Int32},
         Dists_out::AbstractArray{Float32},
         k::Int,
         offset::Int,
         stopP::Int
         )
         #di = [Pair(i, d) for (i,d) in enumerate(Distance[1:stopP,:])]
         di = [Pair(i, d) for (i,d) in enumerate(Distance[1:stopP])]
         
         if size(Indxs_out,1) < k || size(Dists_out,1) < k
             error("The input and output arrays for the fast_sortperm function have less than k dimensions")
         end
         if size(Indxs_out,2) < size(Distance,2) || size(Dists_out,2) < size(Distance,2)
             error("The input and output arrays for the fast_sortperm function have different sizes")
         end
         
         #@inbounds sort!(di, dims=1, by=x->x.second, alg=PartialQuickSort(1:k))
         @inbounds sort!(di, by=x->x.second, alg=PartialQuickSort(1:k))
         
         i = 1
         @inbounds @views for d in di[1:k]
             Indxs_out[i] = d.first + offset
             Dists_out[i] = d.second
             i += 1
         end
         
         di = nothing
         return
end

@inline function getdists!(dist_seg,distance,idxs_seg)
           n = size(dist_seg,2)
           k = size(dist_seg,1)
           
           if size(distance, 2) != n || size(idxs_seg,2) != n
               error("Size mismatch between distance matrixes")
           end
           if size(distance,1) < k 
               error("The number of neighbors is lower than the size of the distance results")
           end
           if size(idxs_seg,1) != k
               error("Size mismatch between distance and index matrixes")
           end
           
           
           @inbounds for i in 1:k
              for j in 1:n
                 dist_seg[i,j] = distance[inxs_seg[i,j],j]
              end
           end

end


@inline function addOffset!(
    idxs::AbstractArray{Int32}, 
    qStart::Int, 
    qStop::Int, 
    cOff::Int
    )
    n = size(idxs, 2)
    k = size(idxs, 1)
    
    pStart = max(1, qStart)
    pStop = min(n, qStop)
    
    if qStart > qStop
       error("Error at adding offset: starting point is greater than ending point")
    end
    
    @inbounds for i in pStart:pStop
        for j in 1:k
           idxs[j,i] = idxs[j,i] + cOff
        end
    end
    
    return
end


@inline function copy2dArr!(
    dest::AbstractArray{T}, 
    source::AbstractArray{T}
    ) where T <: Number
    n = size(source,2)
    d = size(source,1)
    
    if size(dest,2) != n 
       error("Attempting to copy arrays with mismatching number of collumns")
    elseif size(dest,1) != d 
       error("Attempting to copy arrays with mismatching number of rows")
    end
    
    @inbounds for i in 1:n 
       for j in 1:d
          dest[j,i] = source[j,i]
       end
    end
   
   return
end

@inline function copy2dArrPart!(
    dest::AbstractArray{T},
    destStart::Int,
    destEnd::Int,
    source::AbstractArray{T},
    sourceStart::Int,
    sourceEnd::Int
    ) where T <: Number
    nSource = size(source,2)
    dSource = size(source,1)
    
    nDest = size(dest,2)
    dDest = size(dest,1)
    
    if (destEnd - destStart + 1) != (sourceEnd - sourceStart + 1) 
       error("Attempting to copy mismatching number of elements between arrays")
    end
    
    if dSource != dDest 
       error("Attempting to copy arrays with mismatching number of rows")
    end
    
    if destStart > destEnd || sourceStart > sourceEnd
       error("Source and destination marker indexes should be in incrimental order")
    end
    
    if destStart < 1 || sourceStart < 1
       error("Attempting to access elements before an array's start")
    end
    
    if destEnd > nDest || sourceEnd > nSource
      error("Attempting to access elements after an array's end")
    end
    
    n = sourceEnd - sourceStart + 1
    
    #sIndx = sourceStart
    dIndx = destStart
    
    @inbounds for i in sourceStart:sourceEnd
       for j in 1:dSource
          dest[j,dIndx] = source[j,i]
       end
       #sIndx += 1
       dIndx += 1
    end
    
    return
end

@inline function mergeSeg!(
     idx1::AbstractArray{Int32}, 
     dist1::AbstractArray{Float32}, 
     idx2::AbstractArray{Int32}, 
     dist2::AbstractArray{Float32}
     )
     n = size(idx1, 2)
     k = size(idx1, 1)
     
     if n != size(idx2, 2) || n != size(dist1, 2) || n != size(dist2, 2)
        println("Size mismatch for the merging results")
        return
     end
     if k != size(idx2, 1) || k != size(dist1, 1) || k != size(dist2, 1)
        println("Dimension mismatch for the merging results")
        return
     end
     
     Indx = similar(idx1)
     Dists = similar(dist1)
     
     np = Threads.nthreads()
     
     #b = ceil(Int64, n/np)
     
     pt1 = ones(Int64, np)
     pt2 = ones(Int64, np)
    
    #@sync begin    
     #Threads.@threads for p = 1:np
      # @inbounds for i in (p-1)*b + 1 : p*b
     
    @inbounds Threads.@threads for i in 1:n
     # Threads.@spawn begin
        p = Threads.threadid()
        pt1[p] = 1
        pt2[p] = 1
        
           @inbounds for j in 1:k
              if dist1[pt1[p], i] > dist2[pt2[p], i]
                  Dists[j, i] = dist2[pt2[p], i]
                  Indx[j, i] = idx2[pt2[p], i]
               
                  pt2[p] = pt2[p] + 1
              else 
                  Dists[j, i] = dist1[pt1[p], i]
                  Indx[j, i] = idx1[pt1[p], i]
               
                  pt1[p] = pt1[p] + 1           
              end
           
          end
          
          @inbounds @views idx1[1:k,i] .= Indx[1:k,i]
          @inbounds @views dist1[1:k,i] .= Dists[1:k,i]
          
          nothing
    end
     #end
    #end
    
    #@inbounds idx1[:,:] .= Indx[:,:]
    #@inbounds dist1[:,:] .= Dists[:,:]
     
    Indx = nothing
    Dists = nothing
     
    return
end

@inline function mergeSeg!(
     idx1::AbstractArray{Int32}, 
     dist1::AbstractArray{Float32}, 
     idx2::AbstractArray{Int32}, 
     dist2::AbstractArray{Float32},
     Indx::AbstractArray{Int32},
     Dists::AbstractArray{Float32}
     )
     n = size(idx1, 2)
     k = size(idx1, 1)
     
     if n != size(idx2, 2) || n != size(dist1, 2) || n != size(dist2, 2) || n != size(Indx,2) || n != size(Dists, 2)
        println("Size mismatch for the merging results")
        return
     end
     if k != size(idx2, 1) || k != size(dist1, 1) || k != size(dist2, 1) || k != size(Indx,1) || k != size(Dists, 1)
        println("Dimension mismatch for the merging results")
        return
     end

     np = Threads.nthreads()
     
     #b = ceil(Int64, n/np)
     
     pt1 = ones(Int64, np)
     pt2 = ones(Int64, np)
    
    #@sync begin    
     #Threads.@threads for p = 1:np
      # @inbounds for i in (p-1)*b + 1 : p*b
     
    @inbounds Threads.@threads for i in 1:n
        p = Threads.threadid()
        pt1[p] = 1
        pt2[p] = 1
        
           @inbounds for j in 1:k
              if dist1[pt1[p], i] > dist2[pt2[p], i]
                  Dists[j, i] = dist2[pt2[p], i]
                  Indx[j, i] = idx2[pt2[p], i]
               
                  pt2[p] = pt2[p] + 1
              else 
                  Dists[j, i] = dist1[pt1[p], i]
                  Indx[j, i] = idx1[pt1[p], i]
               
                  pt1[p] = pt1[p] + 1           
              end
           
          end
          @inbounds @views idx1[1:k,i] .= Indx[1:k,i]
          @inbounds @views dist1[1:k,i] .= Dists[1:k,i]
          
          nothing
    end
     #end
    #end
    
    #@inbounds idx1[:,:] .= Indx[:,:]
    #@inbounds dist1[:,:] .= Dists[:,:]
    
    return
end

function calc_results!(
        idxs_seg::AbstractArray{Int32},
        dist_seg::AbstractArray{Float32},
        tempIndex::AbstractArray{Int32},
        distance::AbstractArray{Float32},
        c_length::Int,
        q_length::Int,
        k::Int,
        cOff::Int
        )
        
        @views @inbounds Threads.@threads for l in 1:q_length
                          
            idxs_seg[1:(k+1),l] .= partialsortperm!(tempIndex[1:c_length,l],distance[1:c_length,l],1:(k+1),initialized=false) .+ cOff
            dist_seg[1:(k+1),l] .= distance[tempIndex[1:(k+1),l],l]
            
            nothing
       end 
       
       return       
end




@inline function euclDistP(
        Q::AbstractArray{Float32},
        C::AbstractArray{Float32},
        Distance::AbstractArray{Float32},
        pointQ::Int, 
        nC::Int, 
        d::Int
        )
        
        #sQ = @views @inbounds LinearAlgebra.BLAS.dot(d,Q[1:d,pointQ],1,Q[1:d,pointQ],1)
        
        if d > 3
            @inbounds for j in 1:nC
                lastj = d
                
                sum0 = 0.0f0
                sum1 = 0.0f0
                sum2 = 0.0f0
                sum3 = 0.0f0
                sum4 = 0.0f0
                @inbounds for k in 1:4:d-3
                         
                         sum0 += (Q[k,pointQ] - C[k,j])^2
                         sum1 += (Q[k+1,pointQ] - C[k+1,j])^2
                         sum2 += (Q[k+2,pointQ] - C[k+2,j])^2
                         sum3 += (Q[k+3,pointQ] - C[k+3,j])^2
                         
                         lastj = k+3
                end
                @inbounds for k in lastj+1:d
                
                         sum4 += (Q[k,pointQ] - C[k,j])^2
                         
                end
                Distance[j,pointQ] = sum0 + sum1 + sum2 + sum3 +sum4
            end
        else       
            @inbounds for j in 1:nC
                #sC = @views @inbounds LinearAlgebra.BLAS.dot(d,C[1:d,j],1,C[1:d,j],1)
                #@views @inbounds Distance[j,pointQ] = sQ + sC - 2* LinearAlgebra.BLAS.dot(d,Q[1:d,pointQ],1,C[1:d,j],1)
                
                Distance[j,pointQ] = 0.0f0
                
                @inbounds @simd for k in 1:d
                    Distance[j,pointQ] += (Q[k,pointQ] - C[k,j])^2
                end
            end
            
        end
        return
end

@inbounds function euclDist(
        Q::AbstractArray{Float32},
        C::AbstractArray{Float32},
        Distance::AbstractArray{Float32},
        nQ::Int, 
        nC::Int, 
        d::Int
        )
        
        #np = Threads.nthreads()
        
        #tempDist = ones(Float32, np)
        #temp = ones(Float32, np)
        #temp = zeros(Float32, d)
        
         #@inbounds Threads.@threads for i in 1:nQ
         @inbounds for i in 1:nQ
            #p = Threads.threadid()
         
            for j in 1:nC
                #tempDist[p] = 0.0f0
                #temp[p] = 0.0f0
                
                tempDist = 0.0f0
                temp = 0.0f0

                for k in 1:d
                    #temp[p] = Q[k,i] - C[k,j]
                    #tempDist[p] += temp[p]*temp[p]
                    
                    
                    temp = Q[k,i] - C[k,j]
                    tempDist += temp*temp
                end
                #@inbounds Distance[j,i] = tempDist[p]
                @inbounds Distance[j,i] = tempDist
            end
        end
        
        return
end



@inbounds function euclDistGPUb(
            Q_d, 
            C_d, 
            Distance_d, 
            nQ::Int, 
            nC::Int, 
            d::Int
            )
		    xIndex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            yIndex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            
            
            if (xIndex <= nC) && (yIndex <= nQ)
                 tempDist = 0.0f0
                 #temp = 0.0f0
                 
                 @inbounds for j in 1:d
                   #temp = Q_d[j,yIndex] - C_d[j,xIndex]
                   #tempDist += temp*temp
                   tempDist += (Q_d[j,yIndex] - C_d[j,xIndex])^2
                   
                   #if(tempDist > cutoff)
                    #    tempDist = typemax(Float32)
                    #    break
                  # end
                 end
                 @inbounds Distance_d[xIndex,yIndex] = tempDist
			end
            return
end

@inbounds function euclDistGPU(
            Q_d, 
            C_d, 
            Distance_d, 
            nQ::Int, 
            nC::Int, 
            d::Int
            )
		    bIndex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            
            
            @inbounds for index in bIndex:blockDim().x * gridDim().x:nC
   
              @inbounds for i in 1:nQ
               tempDist = 0.0f0
               temp = 0.0f0
               
                @inbounds for j in 1:d
                 temp = Q_d[j,i] - C_d[j,index]
                 tempDist += temp*temp
               end
               Distance_d[index,i] = tempDist
			  end
            end
            return
end

@inline function euclDistGPU_wrapper!(
               Q_d, 
               C_d, 
               Distance_d, 
               nQ::Int, 
               nC::Int, 
               d::Int,
               stopQ::Int,
               stopC::Int,
               process::Int,
               )
               
               NQ = nQ
               NC = nC
               
               if process != 1
                  if stopQ < nQ
                     #NQ = length(1:stopQ)
                     NQ = stopQ
                  end
                  if stopC < nC
                     #NC = length(1:stopC)
                     NC = stopC
                  end
               end
               
               #kernel = @cuda name="euclDistGPU" launch=false euclDistGPU(Q_d, C_d, Distance_d, NQ, NC, d)
               #config = launch_configuration(kernel.fun)
               #threads = Base.min(NC, config.threads)
               #blocks = cld(NC, threads)
               #kernel(Q_d, C_d, Distance_d, NQ, NC, d; threads=threads, blocks=blocks)
               
               kernel = @cuda name="euclDistGPUb" launch=false euclDistGPUb(Q_d, C_d, Distance_d, NQ, NC, d)
               config = launch_configuration(kernel.fun)
               threads = Base.min(NC*NQ, config.threads)
               
               blockSize = floor(Int,sqrt(threads))
               sqBlock = isqrt(blockSize)
               sblX = min(NC, sqBlock)
               sblY = min(NQ, sqBlock)
                
               #nblX = cld(NC, sblX)
               #nblY = cld(NQ, sblY)
               nblX = ceil(Int32, NC/sblX)
               nblY = ceil(Int32, NQ/sblY)
               
               blockSizeRect = (sblX, sblY)
               nblRect = (nblX, nblY)
               
               kernel(Q_d, C_d, Distance_d, NQ, NC, d; threads=blockSizeRect, blocks=nblRect)
                  
               return
end

function GPUlargeKNN(
              Q::AbstractArray{Float32}, 
              C::AbstractArray{Float32}, 
              k::Int
              )
               
              nQ = size(Q,2)
              nC = size(C,2)
              d = size(C,1)
              
              startP = 1
              stopP = nQ
              
              if size(Q,1) != d
                 println("The Query and Corpus arrays have mismatching dimensions")
                 return
              end
               
               tempIndex = zeros(Int64,nC,nQ)
               tempDist = zeros(Float32,nC,nQ)
               
               distance = zeros(Float32,nC,nQ)
               distance_d = CuArray(distance)
               Q_d = CuArray(Q)
               C_d = CuArray(C)
               
               indxs = zeros(Int64,k,nQ)
               dsts = zeros(Float32,k,nQ)
               
               #FullEuclDistGPU_wrapper(X_d,distance_d,startP,stopP,n,d)
               euclDistGPU_wrapper!(Q_d,C_d,distance_d,nQ,nC,d,startP,stopP,1)
               
               copyto!(distance, distance_d)
               
               CUDA.unsafe_free!(distance_d)
               CUDA.unsafe_free!(Q_d)
               CUDA.unsafe_free!(C_d)
               
               #np = Threads.nthreads()
               
			   #b = n ÷ nw
               #b = ceil(Int64, nQ/np)
               #Threads.@threads for p = 1:np
                   #@inbounds for i in (p-1)*b + 1 : p*b
                   @inbounds for i in 1:nQ
                         #tempIndex[1:nC,i] .= 1:nC
                 
                         tempIndex[1:k,i] = fast_sortperm(distance[1:nC,i],k)
                         
                         #partialsortperm!(tempIndex[1:nC,i],distance[1:nC,i],1:k,initialized=true)
                         #tempDist[1:k,i] .= distance[tempIndex[1:k,i],i]
                         for j in 1:k
                             tempDist[j,i] = distance[tempIndex[j,i],i]
                         end
                        
                    end   
               #end
              @inbounds indxs[1:k,1:nQ] .= tempIndex[1:k,1:nQ]
              @inbounds dsts[1:k,1:nQ] .= tempDist[1:k,1:nQ]
               return indxs, dsts
       end




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~                 GPU code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function distPointParKNN_fileout(
             progress::RemoteChannel, 
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int,
             hybrid_all::Bool = true,
             gpuID::Int = 0
             )
               
             if stopP > nQ
                stopP = nQ
             end

             b = stopP - startP + 1
             
             if b < qSizeDef
                qSize = b
             else
                qSize = qSizeDef
             end
             
             device!(gpuID)
             
             filename_indxs = "$(filenameQ).indxs.part$(myid())"
             filename_dists = "$(filenameQ).dists.part$(myid())"
             
             if isfile(filename_indxs)
                 rm(filename_indxs)
             end
             if isfile(filename_dists)
                 rm(filename_dists)
             end
             
             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)
             
               
             idxs = zeros(Int32, k+1, qSize)
             dist = zeros(Float32, k+1, qSize)
      
             idxs_vec = [Vector{Int32}(undef, k+1) for _ in 1:qSize]
             dists_vec = [Vector{Float32}(undef, k+1) for _ in 1:qSize]
    
             idxs_seg = zeros(Int32, k+1, qSize)
             dist_seg = zeros(Float32, k+1, qSize)
             
             Indx = zeros(Int32, k+1, qSize)
             Dists = zeros(Float32, k+1, qSize)

             qStart = startP
             qStop = startP + qSize - 1
             qLocStart = 1
             qLocStop = qSize
             cOff = 0      
             
             @inbounds for i in 1:qBlock  
                load_file_wrapper!(filenameQ,d,Q, qStart:qStop)
                
                cStart = 1
                cStop = cSize
                cOff = cStart - 1  
                
                @inbounds for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1 
            
                   if hybrid_all == true
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1; algorithm=:hybrid_batch_all)                     
                   else
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1; algorithm=:hybrid_batch_test) 
                   end
                   
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:k+1
                             idxs_seg[jj,ii] = idxs_vec[ii][jj] + cOff
                             dist_seg[jj,ii] = dists_vec[ii][jj]
                         end
                   
                   end

                   if j > 1
                      mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                   else
                      copy2dArr!(idxs, idxs_seg)
                      copy2dArr!(dist, dist_seg)
                   end

                   cStart = cStop + 1
                   cStop = min(cStop + cSize, nC)
                   cEnd = cStop - cStart + 1
            
                   
                end
               
               if i == 1
                   if (qLocStop - qLocStart) == (qSize - 1)               
                       store_file(filename_indxs, idxs,false)
                       store_file(filename_dists, dist,false)
                   else
                       store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],false)
                       store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],false)
                   end
               else
                   if (qLocStop - qLocStart) == (qSize - 1)
                       store_file(filename_indxs, idxs,true)
                       store_file(filename_dists, dist,true)  
                   else
                       store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],true)
                       store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],true)
                   end                   
               end
               
               qStart = qStop + 1
               qStop = min(qStop + qSize, nQ)
         
               
               qLocStart = qLocStop + 1
               qLocStop = min(qLocStop + qSize, b)
               qLocStop = min(qLocStop, nQ)
               
               put!(progress, (i, qBlock, gpuID, myid()))
               
               GC.gc()
            end
      
      return (filename_indxs , filename_dists)
end

function distPointParKNN(
             progress::RemoteChannel, 
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int,
             hybrid_all::Bool = true,
             gpuID::Int = 0
             )
               
             if stopP > nQ
                stopP = nQ
             end

             b = stopP - startP + 1
             
             if b < qSizeDef
                qSize = b
             else
                qSize = qSizeDef
             end
             
             device!(gpuID)
             
             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)
             
             
             idxs = zeros(Int32, k+1, qSize)
             dist = zeros(Float32, k+1, qSize)
      
             idxs_seg = similar(idxs)
             dist_seg = similar(dist)
             
             idxs_vec = [Vector{Int32}(undef, k+1) for _ in 1:qSize]
             dists_vec = [Vector{Float32}(undef, k+1) for _ in 1:qSize]
             
             idxs_out = zeros(Int32, k+1, b)
             dist_out = zeros(Float32, k+1, b)
             
             Indx = zeros(Int32, k+1, qSize)
             Dists = zeros(Float32, k+1, qSize)
             
             qStart = startP
             qStop = startP + qSize - 1
             qLocStart = 1
             qLocStop = qSize
             cOff = 0      
             
             @inbounds for i in 1:qBlock  
                load_file_wrapper!(filenameQ,d,Q, qStart:qStop)
                
                cStart = 1
                cStop = cSize
                cOff = cStart - 1  
                
                @inbounds for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1 
                   
                   if hybrid_all == true
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1; algorithm=:hybrid_batch_all)                     
                   else
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1; algorithm=:hybrid_batch_test) 
                   end
                   
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:k+1
                             idxs_seg[jj,ii] = idxs_vec[ii][jj] + cOff
                             dist_seg[jj,ii] = dists_vec[ii][jj]
                         end
                   
                   end
            
                   #addOffset!(idxs_seg, 1, length(qStart:qStop), cOff)
            
                   if j > 1
                      mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                   else
                      copy2dArr!(idxs, idxs_seg)
                      copy2dArr!(dist, dist_seg)
                   end

                   cStart = cStop + 1
                   cStop = min(cStop + cSize, nC)
                   cEnd = cStop - cStart + 1
            
                   
                end
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:(k+1)
                            idxs_out[w,z] = idxs[w,lcount]
                            dist_out[w,z] = dist[w,lcount]
                         end
                     
                         lcount += 1
                     end
               end
               
               qStart = qStop + 1
               qStop = min(qStop + qSize, nQ)
         
               
               qLocStart = qLocStop + 1
               qLocStop = min(qLocStop + qSize, b)
               qLocStop = min(qLocStop, nQ)
               
               put!(progress, (i, qBlock, gpuID, myid()))
               
               GC.gc()
            end            
      
      return (idxs_out , dist_out)
end


function runHiggsKNN_GPU(
      filename_input::String,
      k::Int, 
      d::Int = 28, 
      n::Int = 11000000, 
      cSize::Int = 100000, 
      qSize::Int = 11000
      )
      
      data = load_file(filename_input, d, n)
      n = size(data, 2)
      d = size(data, 1)

      
      C = zeros(Float32, d, cSize)
      Q = zeros(Float32, d, qSize)
       
      idxs = zeros(Int64, k+1, qSize)
      dist = zeros(Float32, k+1, qSize)     
      
      idxs_seg = zeros(Int64, k+1, qSize)
      dist_seg = zeros(Float32, k+1, qSize)
      
      idxs_out = zeros(Int64, k+1, n)
      dist_out = zeros(Float32, k+1, n)
      
      cStart = 1
      cStop = cSize
      cEnd = cSize
      
      qStart = 1
      qStop = qSize
      qElapsed = 1
      cOff = 0      
      println("qStart $qStart , qStop $qStop , qBlock $qBlock , Q $(size(Q,1) ) x $(size(Q,2) )")
     
      #@inbounds for i in 1:qBlock
      for i in 1:qBlock
         Q[:,1:qSize] .= data[:,qStart:qStop]
            
         cStart = 1
         cStop = cSize
         cOff = cStart - 1         
         
         for j in 1:cBlock
            C[:,1:cSize] .= data[:,cStart:cStop]
            
            cOff = cStart - 1 
            
            idxs_seg[:,1:qSize], dist_seg[:,1:qSize] = GPUlargeKNN(Q, C, k+1);
           
            
            addOffset!(idxs_seg, 1, qSize, cOff)
            
            #println("qStart $qStart, qStop $qStop, cOff $cOff")
            
            if j > 1
               mergeSeg!(idxs, dist, idxs_seg, dist_seg)
            else
               idxs[:,:] .= idxs_seg[:,:]
               dist[:,:] .= dist_seg[:,:]
            end
            
            cStart = cStop + 1
            cStop = min(cStop + cSize, n)
            cEnd = cStop - cStart + 1
            
            
         end
         
        @views idxs_out[:,qStart:qStop] .= idxs[:,1:qSize]
        @views dist_out[:,qStart:qStop] .= dist[:,1:qSize]
         
         println("Done $i segments out of $qBlock")
         qStart = qStop + 1
         qStop = min(qStop + qSize, n)
         
      end
      
      return (idxs_out , dist_out)
end

function distPointKNN_GPU(
             progress::RemoteChannel,
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int,
             gpuID::Int = 0
             )
               
             if stopP > nQ
                stopP = nQ
             end

             b = stopP - startP + 1
             
             if b < qSizeDef
                qSize = b
             else
                qSize = qSizeDef
             end
             
             device!(gpuID)
             
             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = CUDA.Mem.pin(zeros(Float32, d, cSize))
             Q = CUDA.Mem.pin(zeros(Float32, d, qSize))
             
             C_d = CuArray(zeros(Float32, d, cSize))
             Q_d = CuArray(zeros(Float32, d, qSize))
             
             
             idxs = zeros(Int32, k+1, qSize)
             dist = zeros(Float32, k+1, qSize)
      
             idxs_seg = zeros(Int32, k+1, qSize)
             dist_seg = zeros(Float32, k+1, qSize)
             
             Indx = zeros(Int32, k+1, qSize)
             Dists = zeros(Float32, k+1, qSize)
             
             idxs_out = zeros(Int32, k+1, b)
             dist_out = zeros(Float32, k+1, b)
             
             distance = CUDA.Mem.pin(zeros(Float32,cSize,qSize))
             
             tempIndex = zeros(Int32,cSize,qSize)
             
             
             distance_d_even = CuArray(distance)
             distance_d_odd = CuArray(distance)
             
             
             np = Threads.nthreads()
             th = ceil(Int32, qSize/np)
            
             sort_finished = @async nothing
             merge_needed = false
            
             qStart = startP
             qStop = startP + qSize - 1
             qLocStart = 1
             qLocStop = qSize
             cOff = 0      
             
             @inbounds for i in 1:qBlock  
                load_file_wrapper!(filenameQ,d,Q, qStart:qStop)

                copyto!(Q_d, Q)
                
                cStart = 1
                cStop = cSize
                cOff = cStart - 1  
                
                @inbounds for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
                   
                   copyto!(C_d, C)
                   
                   if iseven(j)
                        CUDA.@sync euclDistGPU_wrapper!(Q_d,C_d,distance_d_even,qSize,cSize,d,length(qStart:qStop),length(cStart:cStop),2)                        
                   else
                        CUDA.@sync euclDistGPU_wrapper!(Q_d,C_d,distance_d_odd,qSize,cSize,d,length(qStart:qStop),length(cStart:cStop),2)
                   end
                   
                   wait(sort_finished)
                   
                   cOff = cStart - 1
                   q_length = min(qSize, length(qStart:qStop)) 
                   c_length = min(cSize, length(cStart:cStop))
                   
                   merge_needed = j > 1 ? true : false
                   
                   distDone = iseven(j) ? 0 : 1 
                   
                   sort_finished = @async begin
                       if distDone == 0
                            copyto!(distance, distance_d_even)
                       else
                            copyto!(distance, distance_d_odd)
                       end
                       
                       calc_results!(idxs_seg, dist_seg, tempIndex, distance, c_length, q_length, k, cOff)
                
                       if merge_needed
                          mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                       else
                          copy2dArr!(idxs, idxs_seg)
                          copy2dArr!(dist, dist_seg)
                       end
                       
                       nothing
                   end

                   cStart = cStop + 1
                   cStop = min(cStop + cSize, nC)
                   cEnd = cStop - cStart + 1
            
                   
                end
                
               wait(sort_finished)
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   #copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   #copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
                   
                   @views @inbounds idxs_out[1:(k+1),qLocStart:qLocStop] .= idxs[1:(k+1),1:qSize]
                   @views @inbounds dist_out[1:(k+1),qLocStart:qLocStop] .= dist[1:(k+1),1:qSize]
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:(k+1)
                            idxs_out[w,z] = idxs[w,lcount]
                            dist_out[w,z] = dist[w,lcount]
                         end
                     
                         lcount += 1
                     end
               end
               
               qStart = qStop + 1
               qStop = min(qStop + qSize, nQ)
         
               
               qLocStart = qLocStop + 1
               qLocStop = min(qLocStop + qSize, b)
               qLocStop = min(qLocStop, nQ)
               
               put!(progress, (i, qBlock, gpuID, myid()))
               
               GC.gc()
            end            
      
      CUDA.unsafe_free!(Q_d)
      CUDA.unsafe_free!(C_d)
      CUDA.unsafe_free!(distance_d_even)
      CUDA.unsafe_free!(distance_d_odd)
      
      return (idxs_out , dist_out)
end

function distPointKNN_GPU_fileout(
             progress::RemoteChannel,
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int,
             gpuID::Int = 0
             )
               
         if stopP > nQ
            stopP = nQ
         end

         b = stopP - startP + 1
         
         if b < qSizeDef
                qSize = b
         else
                qSize = qSizeDef
         end
         
         filename_indxs = "$(filenameQ).indxs.part$(myid())"
         filename_dists = "$(filenameQ).dists.part$(myid())"
         
         if isfile(filename_indxs)
             rm(filename_indxs)
         end
         if isfile(filename_dists)
             rm(filename_dists)
         end
         
         device!(gpuID)
         
         cBlock = ceil(Int32, nC/cSize)
         qBlock = ceil(Int32, b/qSize)
          
         C = CUDA.Mem.pin(zeros(Float32, d, cSize))
         Q = CUDA.Mem.pin(zeros(Float32, d, qSize))
           
         C_d = CuArray(zeros(Float32, d, cSize))
         Q_d = CuArray(zeros(Float32, d, qSize))

         idxs = zeros(Int32, k+1, qSize)
         dist = zeros(Float32, k+1, qSize)
  
         idxs_seg = zeros(Int32, k+1, qSize)
         dist_seg = zeros(Float32, k+1, qSize)
         
         Indx = zeros(Int32, k+1, qSize)
         Dists = zeros(Float32, k+1, qSize)
         
         distance = CUDA.Mem.pin(zeros(Float32,cSize,qSize))
         
         tempIndex = zeros(Int32,cSize,qSize)
         
         distance_d_even = CuArray(distance)
         distance_d_odd = CuArray(distance)

         np = Threads.nthreads()
         th = ceil(Int32, qSize/np)
         
         
         nw = nworkers()
         wks = collect(workers())
         placeholders = []

         sort_finished = @async nothing
         merge_needed = false
        
         qStart = startP
         qStop = min(startP + qSize - 1, stopP)
         qLocStart = 1
         qLocStop = qSize
         cOff = 0      
         
         @inbounds for i in 1:qBlock  
            load_file_wrapper!(filenameQ,d,Q, qStart:qStop)
            copyto!(Q_d, Q)
            
            cStart = 1
            cStop = min(cSize, stopP)
            cOff = cStart - 1  
            
            @inbounds for j in 1:cBlock
               load_file_wrapper!(filenameC,d,C, cStart:cStop)
               copyto!(C_d, C)
               
               if iseven(j)
                    CUDA.@sync euclDistGPU_wrapper!(Q_d,C_d,distance_d_even,qSize,cSize,d,length(qStart:qStop),length(cStart:cStop),2)                        
               else
                    CUDA.@sync euclDistGPU_wrapper!(Q_d,C_d,distance_d_odd,qSize,cSize,d,length(qStart:qStop),length(cStart:cStop),2)
               end
               
               wait(sort_finished)
               
               cOff = cStart - 1
               sort_length = length(cStart:cStop)
               q_length = min(qSize, (qStop - qStart +1))
               c_length = min(cSize, length(cStart:cStop))               
               
               merge_needed = j > 1 ? true : false
               
               distDone = iseven(j) ? 0 : 1
               
               
               sort_finished = @async begin
                   if distDone == 0
                        copyto!(distance, distance_d_even)
                   else
                        copyto!(distance, distance_d_odd)
                   end
                   
                   calc_results!(idxs_seg, dist_seg, tempIndex, distance, c_length, q_length, k, cOff)
            
                   if merge_needed
                      mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                   else
                      copy2dArr!(idxs, idxs_seg)
                      copy2dArr!(dist, dist_seg)
                   end
                   nothing
               end

               cStart = cStop + 1
               cStop = min(cStop + cSize, nC)
               cEnd = cStop - cStart + 1
        
               
            end
           
           wait(sort_finished)
           
           if i == 1
               if (qLocStop - qLocStart) == (qSize - 1)               
                   store_file(filename_indxs, idxs,false)
                   store_file(filename_dists, dist,false)
               else
                   store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],false)
                   store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],false)
               end
           else
               if (qLocStop - qLocStart) == (qSize - 1)
                   store_file(filename_indxs, idxs,true)
                   store_file(filename_dists, dist,true)  
               else
                   store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],true)
                   store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],true)
               end                   
           end
           
           qStart = qStop + 1
           qStop = min(qStop + qSize, nQ)
     
           
           qLocStart = qLocStop + 1
           qLocStop = min(qLocStop + qSize, b)
           qLocStop = min(qLocStop, nQ)
           
           put!(progress, (i, qBlock, gpuID, myid()))
           
           GC.gc()
        end               
  
        CUDA.unsafe_free!(Q_d)
        CUDA.unsafe_free!(C_d)
        CUDA.unsafe_free!(distance_d_even)
        CUDA.unsafe_free!(distance_d_odd)
      
      return (filename_indxs , filename_dists)
end




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~               CPU only code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function distPointKNN_fileout(
             progress::RemoteChannel, 
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int,
             bf::Bool = true
             )
               
             if stopP > nQ
                stopP = nQ
             end

             b = stopP - startP + 1
             
             if b < qSizeDef
                qSize = b
             else
                qSize = qSizeDef
             end
             
             filename_indxs = "$(filenameQ).indxs.part$(myid())"
             filename_dists = "$(filenameQ).dists.part$(myid())"
             
             if isfile(filename_indxs)
                 rm(filename_indxs)
             end
             if isfile(filename_dists)
                 rm(filename_dists)
             end
             
             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)
               
             idxs = zeros(Int32, k+1, qSize)
             dist = zeros(Float32, k+1, qSize)
      
             idxs_seg = zeros(Int32, k+1, qSize)
             dist_seg = zeros(Float32, k+1, qSize)
             
             Indx = zeros(Int32, k+1, qSize)
             Dists = zeros(Float32, k+1, qSize)
            
             qStart = startP
             qStop = startP + qSize - 1
             qLocStart = 1
             qLocStop = qSize
             cOff = 0      
             
             @inbounds for i in 1:qBlock  
                load_file_wrapper!(filenameQ,d,Q, qStart:qStop)
                
                cStart = 1
                cStop = cSize
                cOff = cStart - 1  
                
                @inbounds for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1 
            
                   if bf == true
                       knn!(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1, FLANNParameters(algorithm=0), idxs_seg, dist_seg);
                   else
                       knn!(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1, FLANNParameters(algorithm=4, trees=16,leaf_max_size=64,branching=32,iterations=14,centers_init=0,cb_index=0.2), idxs_seg, dist_seg);
                   end
           
            
                   addOffset!(idxs_seg, 1, length(qStart:qStop), cOff)
            
                   if j > 1
                      mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                   else
                      copy2dArr!(idxs, idxs_seg)
                      copy2dArr!(dist, dist_seg)
                   end

                   cStart = cStop + 1
                   cStop = min(cStop + cSize, nC)
                   cEnd = cStop - cStart + 1
            
                   
                end
               
               if i == 1
                   if (qLocStop - qLocStart) == (qSize - 1)               
                       store_file(filename_indxs, idxs,false)
                       store_file(filename_dists, dist,false)
                   else
                       store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],false)
                       store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],false)
                   end
               else
                   if (qLocStop - qLocStart) == (qSize - 1)
                       store_file(filename_indxs, idxs,true)
                       store_file(filename_dists, dist,true)  
                   else
                       store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],true)
                       store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],true)
                   end                   
               end
               
               qStart = qStop + 1
               qStop = min(qStop + qSize, nQ)
         
               
               qLocStart = qLocStop + 1
               qLocStop = min(qLocStop + qSize, b)
               qLocStop = min(qLocStop, nQ)
               
               put!(progress, (i, qBlock, myid()))
               
               GC.gc()
            end
            
            C = nothing
            Q = nothing
            
            idxs = nothing
            dist = nothing
      
            idxs_seg = nothing
            dist_seg = nothing
             
            Indx = nothing
            Dists = nothing
      
      return (filename_indxs , filename_dists)
end

function distPointKNN(
             progress::RemoteChannel, 
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int,
             bf::Bool = true
             )
               
             if stopP > nQ
                stopP = nQ
             end

             b = stopP - startP + 1
             
             if b < qSizeDef
                qSize = b
             else
                qSize = qSizeDef
             end
             
             
             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)

             idxs = zeros(Int32, k+1, qSize)
             dist = zeros(Float32, k+1, qSize)
      
             idxs_seg = similar(idxs)
             dist_seg = similar(dist)
             
             
             idxs_out = zeros(Int32, k+1, b)
             dist_out = zeros(Float32, k+1, b)
             
             Indx = zeros(Int32, k+1, qSize)
             Dists = zeros(Float32, k+1, qSize)
            
             qStart = startP
             qStop = startP + qSize - 1
             qLocStart = 1
             qLocStop = qSize
             cOff = 0      
             
             @inbounds for i in 1:qBlock  
                load_file_wrapper!(filenameQ,d,Q, qStart:qStop)
                
                cStart = 1
                cStop = cSize
                cOff = cStart - 1  
                
                @inbounds for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1 
                   
                   if bf == true
                       knn!(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1, FLANNParameters(algorithm=0), idxs_seg, dist_seg);
                   else
                       knn!(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1, FLANNParameters(algorithm=4, trees=16,leaf_max_size=64,branching=32,iterations=14,centers_init=0,cb_index=0.2), idxs_seg, dist_seg); 
                   end
           
            
                   addOffset!(idxs_seg, 1, length(qStart:qStop), cOff)
            
                   if j > 1
                      mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                   else
                      copy2dArr!(idxs, idxs_seg)
                      copy2dArr!(dist, dist_seg)
                   end

                   cStart = cStop + 1
                   cStop = min(cStop + cSize, nC)
                   cEnd = cStop - cStart + 1
            
                   
                end
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:(k+1)
                            idxs_out[w,z] = idxs[w,lcount]
                            dist_out[w,z] = dist[w,lcount]
                         end
                     
                         lcount += 1
                     end
               end
               
               qStart = qStop + 1
               qStop = min(qStop + qSize, nQ)
         
               
               qLocStart = qLocStop + 1
               qLocStop = min(qLocStop + qSize, b)
               qLocStop = min(qLocStop, nQ)
               
               put!(progress, (i, qBlock, myid()))
               
               GC.gc()
            end

            C = nothing
            Q = nothing
            
            idxs = nothing
            dist = nothing
      
            idxs_seg = nothing
            dist_seg = nothing
             
            Indx = nothing
            Dists = nothing            
      
      return (idxs_out , dist_out)
end

function runHiggsKNN(
      filename_input::String,
      k::Int, 
      d::Int = 28, 
      n::Int = 11000000, 
      cSize::Int = 100000, 
      qSize::Int = 11000, 
      bf::Bool = true
      )

      cBlock = ceil(Int32, n/cSize)
      qBlock = ceil(Int32, n/qSize)
      
      C = zeros(Float32, d, cSize)
      Q = zeros(Float32, d, qSize)
      
      idxs = zeros(Int32, k+1, qSize)
      dist = zeros(Float32, k+1, qSize)
      
      idxs_seg = similar(idxs)
      dist_seg = similar(dist)
      
      idxs_out = zeros(Int32, k+1, n)
      dist_out = zeros(Float32, k+1, n)
      
      cStart = 1
      cStop = cSize
      cEnd = cSize
      
      qStart = 1
      qStop = qSize
      qLocStart = 1
      qLocStop = qSize
      cOff = 0      
      println("qStart $qStart , qStop $qStop , qBlock $qBlock , Q $(size(Q,1) ) x $(size(Q,2) )")
     
      for i in 1:qBlock    
         load_file_wrapper!(filename_input,d,Q, qStart:qStop)
            
         cStart = 1
         cStop = cSize
         cOff = cStart - 1         
         
         for j in 1:cBlock
            load_file_wrapper!(filename_input,d,C, cStart:cStop)
            
            cOff = cStart - 1 
            
            if bf == true
               knn!(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1, FLANNParameters(algorithm=0), idxs_seg, dist_seg);
            else
               knn!(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], k+1, FLANNParameters(algorithm=4, trees=16,leaf_max_size=64,branching=32,iterations=14,centers_init=0,cb_index=0.2), idxs_seg, dist_seg);
            end
           
            
            addOffset!(idxs_seg, 1, qSize, cOff)
            
            if j > 1
              mergeSeg!(idxs, dist, idxs_seg, dist_seg)
            else
              copy2dArr!(idxs, idxs_seg)
              copy2dArr!(dist, dist_seg)
            end
            
            cStart = cStop + 1
            cStop = min(cStop + cSize, n)
            cEnd = cStop - cStart + 1
       
         end
         
         if (qLocStop - qLocStart) == (qSize - 1)
            copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
            copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
         else
            lcount = 1
            @inbounds for z in qLocStart:qLocStop
                         for w in 1:(k+1)
                            idxs_out[w,z] = idxs[w,lcount]
                            dist_out[w,z] = dist[w,lcount]
                         end
                         lcount += 1
                      end
         end
       
         println("Done $i segments out of $qBlock")
         qStart = qStop + 1
         qStop = min(qStop + qSize, n)
         
         qLocStart = qLocStop + 1
         qLocStop = min(qLocStop + qSize, n)
         
         GC.gc()
         
      end
      
      return (idxs_out , dist_out)
end



function runHiggsKNN_CPU(
      filename_input::String,
      k::Int, 
      d::Int = 28, 
      n::Int = 11000000, 
      cSize::Int = 100000, 
      qSize::Int = 11000
      )
      
      data = load_file(filename_input, d, n)
      n = size(data, 2)
      d = size(data, 1)

      cBlock = ceil(Int64, n/cSize)
      qBlock = ceil(Int64, n/qSize)
      
      C = zeros(Float32, d, cSize)
      Q = zeros(Float32, d, qSize)
     
      idxs = zeros(Int64, k+1, qSize)
      dist = zeros(Float32, k+1, qSize)
      
      idxs_seg = zeros(Int64, k+1, qSize)
      dist_seg = zeros(Float32, k+1, qSize)
      
      idxs_out = zeros(Int64, k+1, b)
      dist_out = zeros(Float32, k+1, b)
      
      cStart = 1
      cStop = cSize
      cEnd = cSize
      
      qStart = 1
      qStop = qSize
      qElapsed = 1
      cOff = 0      
      println("qStart $qStart , qStop $qStop , qBlock $qBlock , Q $(size(Q,1) ) x $(size(Q,2) )")
     
      #@inbounds for i in 1:qBlock
      for i in 1:qBlock
         Q[:,1:qSize] .= data[:,qStart:qStop]
            
         cStart = 1
         cStop = cSize
         cOff = cStart - 1         
         
         for j in 1:cBlock
            C[:,1:cSize] .= data[:,cStart:cStop]
            
            cOff = cStart - 1 
            
            idxs_seg[:,1:qSize], dist_seg[:,1:qSize] = GPUlargeKNN(Q, C, k+1);
           
            
            addOffset!(idxs_seg, 1, qSize, cOff)
            
            #println("qStart $qStart, qStop $qStop, cOff $cOff")
            
            if j > 1
               mergeSeg!(idxs, dist, idxs_seg, dist_seg)
            else
               idxs[:,:] .= idxs_seg[:,:]
               dist[:,:] .= dist_seg[:,:]
            end
            
            cStart = cStop + 1
            cStop = min(cStop + cSize, n)
            cEnd = cStop - cStart + 1
            
            
         end
         
        @views idxs_out[:,qStart:qStop] .= idxs[:,1:qSize]
        @views dist_out[:,qStart:qStop] .= dist[:,1:qSize]
         
         println("Done $i segments out of $qBlock")
         qStart = qStop + 1
         qStop = min(qStop + qSize, n)
         
      end
      
      return (idxs_out , dist_out)
      #return (idxs , dist)

end

function distPointKNN_CPU(
             progress::RemoteChannel, 
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int
             )
               
             if stopP > nQ
                stopP = nQ
             end

             b = stopP - startP + 1
             
             if b < qSizeDef
                qSize = b
             else
                qSize = qSizeDef
             end
             
             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)
             
             idxs = zeros(Int32, k+1, qSize)
             dist = zeros(Float32, k+1, qSize)
      
             idxs_seg = zeros(Int32, k+1, qSize)
             dist_seg = zeros(Float32, k+1, qSize)
             
             Indx = zeros(Int32, k+1, qSize)
             Dists = zeros(Float32, k+1, qSize)
             
     
             idxs_out = zeros(Int32, k+1, b)
             dist_out = zeros(Float32, k+1, b)
             
             distance = zeros(Float32,cSize,qSize)
             
             tempIndex = zeros(Int32,cSize,qSize)
             
             
             np = Threads.nthreads()
             #th = qSize ÷ np
             th = ceil(Int32, qSize/np)
            
             merge_needed = false
            
             qStart = startP
             qStop = startP + qSize - 1
             qLocStart = 1
             qLocStop = qSize
             cOff = 0      
             
             @inbounds for i in 1:qBlock  
                load_file_wrapper!(filenameQ,d,Q, qStart:qStop)

                
                cStart = 1
                cStop = cSize
                cOff = cStart - 1  
                
                @inbounds for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)

                   cOff = cStart - 1 
                   
                   q_length = min(qSize, length(qStart:qStop)) 
                   c_length = min(cSize, length(cStart:cStop))

                   @views @inbounds Threads.@threads for l in 1:q_length
                        euclDistP(Q,C,distance,l,c_length,d)                   
                        idxs_seg[1:(k+1),l] .= partialsortperm!(tempIndex[1:c_length,l],distance[1:c_length,l],1:(k+1),initialized=false) .+ cOff
                        dist_seg[1:(k+1),l] .= distance[tempIndex[1:(k+1),l],l]
                        
                        nothing
                   end  
                   
                   merge_needed = j > 1 ? true : false
                   if merge_needed
                      mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                   else
                      copy2dArr!(idxs, idxs_seg)
                      copy2dArr!(dist, dist_seg)
                   end
                       

                   cStart = cStop + 1
                   cStop = min(cStop + cSize, nC)
                   cEnd = cStop - cStart + 1
            
                   
                end
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
                   
                   #@views @inbounds idxs_out[1:(k+1),qLocStart:qLocStop] .= idxs[1:(k+1),1:qSize]
                   #@views @inbounds dist_out[1:(k+1),qLocStart:qLocStop] .= dist[1:(k+1),1:qSize]
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:(k+1)
                            idxs_out[w,z] = idxs[w,lcount]
                            dist_out[w,z] = dist[w,lcount]
                         end
                     
                         lcount += 1
                     end
               end
               
               qStart = qStop + 1
               qStop = min(qStop + qSize, nQ)
         
               
               qLocStart = qLocStop + 1
               qLocStop = min(qLocStop + qSize, b)
               qLocStop = min(qLocStop, nQ)
               
               put!(progress, (i, qBlock, myid()))
               
               GC.gc()
            end
            
            C = nothing
            Q = nothing
             
            idxs = nothing
            dist = nothing
      
            idxs_seg = nothing
            dist_seg = nothing
             
            Indx = nothing
            Dists = nothing
            
            distance = nothing
             
            tempIndex = nothing           
      
      return (idxs_out , dist_out)
end

function distPointKNN_CPU_fileout(
             progress::RemoteChannel, 
             filenameC::String,             
             filenameQ::String, 
             startP::Int, 
             stopP::Int, 
             cSize::Int, 
             qSizeDef::Int, 
             nC::Int, 
             nQ::Int, 
             d::Int, 
             k::Int
             )
               
         if stopP > nQ
            stopP = nQ
         end

         b = stopP - startP + 1
         
         if b < qSizeDef
                qSize = b
         else
                qSize = qSizeDef
         end
         
         filename_indxs = "$(filenameQ).indxs.part$(myid())"
         filename_dists = "$(filenameQ).dists.part$(myid())"
         
         if isfile(filename_indxs)
             rm(filename_indxs)
         end
         if isfile(filename_dists)
             rm(filename_dists)
         end
         
         cBlock = ceil(Int32, nC/cSize)
         qBlock = ceil(Int32, b/qSize)
          
         C = zeros(Float32, d, cSize)
         Q = zeros(Float32, d, qSize)
           
         idxs = zeros(Int32, k+1, qSize)
         dist = zeros(Float32, k+1, qSize)
         
         idxs_seg = zeros(Int32, k+1, qSize)
         dist_seg = zeros(Float32, k+1, qSize)
  
         Indx = zeros(Int32, k+1, qSize)
         Dists = zeros(Float32, k+1, qSize)
         
         distance = zeros(Float32,cSize,qSize)
         
         tempIndex = zeros(Int32,cSize,qSize) 
         
         np = Threads.nthreads()
         #th = qSize ÷ np
         th = ceil(Int32, qSize/np)
         
         nw = nworkers()
         wks = collect(workers())
         placeholders = []

         merge_needed = false
        
         qStart = startP
         qStop = min(startP + qSize - 1, stopP)
         qLocStart = 1
         qLocStop = qSize
         cOff = 0      
         
         @inbounds for i in 1:qBlock  
            load_file_wrapper!(filenameQ,d,Q, qStart:qStop)
            
            cStart = 1
            cStop = min(cSize, stopP)
            cOff = cStart - 1  
            
            @inbounds for j in 1:cBlock
               load_file_wrapper!(filenameC,d,C, cStart:cStop)
        
               #euclDist(Q,C,distance,length(qStart:qStop),length(cStart:cStop),d)
               
               cOff = cStart - 1 
               
               q_length = min(qSize, length(qStart:qStop)) 
               c_length = min(cSize, length(cStart:cStop))               

               @views @inbounds Threads.@threads for l in 1:q_length
                    euclDistP(Q,C,distance,l,c_length,d)                   
                    idxs_seg[1:(k+1),l] .= partialsortperm!(tempIndex[1:c_length,l],distance[1:c_length,l],1:(k+1),initialized=false) .+ cOff
                    dist_seg[1:(k+1),l] .= distance[tempIndex[1:(k+1),l],l]
                    
                    nothing
               end  

               merge_needed = j > 1 ? true : false        
               if merge_needed
                  mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                  #mergeSeg!(idxs, dist, idxs_seg, dist_seg)
               else
                  copy2dArr!(idxs, idxs_seg)
                  copy2dArr!(dist, dist_seg)
               end

               cStart = cStop + 1
               cStop = min(cStop + cSize, nC)
               cEnd = cStop - cStart + 1
        
               
            end
           
           if i == 1
               if (qLocStop - qLocStart) == (qSize - 1)               
                   store_file(filename_indxs, idxs,false)
                   store_file(filename_dists, dist,false)
               else
                   store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],false)
                   store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],false)
               end
           else
               if (qLocStop - qLocStart) == (qSize - 1)
                   store_file(filename_indxs, idxs,true)
                   store_file(filename_dists, dist,true)  
               else
                   store_file(filename_indxs, idxs[:,1:(qLocStop-qLocStart+1)],true)
                   store_file(filename_dists, dist[:,1:(qLocStop-qLocStart+1)],true)
               end                   
           end
           
           qStart = qStop + 1
           qStop = min(qStop + qSize, nQ)
     
           
           qLocStart = qLocStop + 1
           qLocStop = min(qLocStop + qSize, b)
           qLocStop = min(qLocStop, nQ)
           
           put!(progress, (i, qBlock, myid()))
           
           GC.gc()
        end

        C = nothing
        Q = nothing
           
        idxs = nothing
        dist = nothing
         
        idxs_seg = nothing
        dist_seg = nothing
  
        Indx = nothing
        Dists = nothing
         
        distance = nothing
         
        tempIndex = nothing         
      
      return (filename_indxs , filename_dists)
end




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~               Main code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function distributedKNN(
      filenameC::String,
      filenameQ::String,
      k::Int, 
      d::Int = 28, 
      nC::Int = 11000000,
      nQ::Int = 11000000,      
      cSize::Int = 100000, 
      qSize::Int = 11000,
      algorithm::Int = 3,    # Algorithm 0 == FLANN (algorithm 0)
                            # Algorithm 1 == FLANN (approximated)
                            # Algorithm 2 == GPU
                            # Algorithm 3 == CPU
                            # Algorithm 4 == ParallelNeighbors
      in_memory::Bool = false
      )
      
      nearestKN = zeros(Int32,k+1,nQ)
      distsKN =  zeros(Float32,k+1,nQ)
  
      nw = nworkers()
      np = Threads.nthreads()        
      
      b = ceil(Int64, nQ/nw)

      idxs_seg = zeros(Int32, k+1,b)
      dist_seg = zeros(Float32, k+1, b)
      
      placeholders = []
      
      numJobs = ceil(Int32, nQ/qSize)
      progress = RemoteChannel(()->Channel{Tuple}(numJobs));
      
      GPUs = Int(length(devices()))
      gpuID = 0
      
      println("Starting distributed kNN calculation of $nQ Query points from file \"$filenameQ\" and $nC Corpus points from file \"$filenameC\"")
      println(" Number of workers: $nw")
      println(" Number of threads: $np")
      if algorithm == 2 || algorithm == 4
         
         println(" Number of GPUs: $GPUs")
      end
      println(" Total number of job segments: $numJobs")
      
      #@sync begin
      jobCount = numJobs
      
      startP = 1
      stopP = b
      for j in workers()
         if in_memory == false
               if algorithm == 0
                     pl = @spawnat j distPointKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true)
                     jobCount -= 1
                elseif algorithm == 1 
                     pl = @spawnat j distPointKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, false)
                     jobCount -= 1
                elseif algorithm == 2
                     pl = @spawnat j distPointKNN_GPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     gpuID = (gpuID + 1) % GPUs 
                     jobCount -= 1
                elseif algorithm ==3
                     pl = @spawnat j distPointKNN_CPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k) 
                     jobCount -= 1
                else
                     pl = @spawnat j distPointParKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, false, gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                end
         else
               if algorithm == 0
                     pl = @spawnat j distPointKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true)
                     jobCount -= 1
                elseif algorithm == 1 
                     pl = @spawnat j distPointKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, false)
                     jobCount -= 1
                elseif algorithm == 2
                     pl = @spawnat j distPointKNN_GPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     gpuID = (gpuID + 1) % GPUs 
                     jobCount -= 1
                elseif algorithm ==3
                     pl = @spawnat j distPointKNN_CPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k) 
                     jobCount -= 1
                else
                     pl = @spawnat j distPointParKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, false,gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                end
         end
         
         push!(placeholders,pl)
         
         if jobCount == 0 
             break
         end
         
         startP = stopP + 1
         stopP = min(stopP + b, nQ)
      end
       #end
       
       jobCount = numJobs
       counter = 0
       if algorithm == 2 || algorithm == 4
           while jobCount > 0
              job_id, local_jobs, gpu_ID, myid = take!(progress)
              counter += 1
              println("  $counter of $numJobs done: Worker $myid finished local job $job_id of $local_jobs on GPU $gpu_ID")
              jobCount -= 1
           end
       else
           while jobCount > 0
              job_id, local_jobs, myid = take!(progress)
              counter += 1
              println("  $counter of $numJobs done: Worker $myid finished local job $job_id of $local_jobs")
              jobCount -= 1
           end
       end
       
       jobCount = numJobs
       
       startP = 1
       stopP = b
       if in_memory == false
           @inbounds for j in workers()
               filename_indxs, filename_dists = fetch(popfirst!(placeholders))
               #try
               #     filename_indxs, filename_dists = fetch(popfirst!(placeholders))
               #catch ex
               #      println(ex)
               #end
               
               jobCount -= 1
               
               load_bin_segment_file!(filename_indxs, k+1, idxs_seg, 1:b, Int32)
               load_bin_segment_file!(filename_dists, k+1, dist_seg, 1:b, Float32)
               
               if isfile(filename_indxs)
                  rm(filename_indxs)
               end
               if isfile(filename_dists)
                  rm(filename_dists)
               end
               
               nearestKN[1:k+1,startP:stopP] .= idxs_seg[1:k+1,:]
               distsKN[1:k+1,startP:stopP] .= dist_seg[1:k+1,:]
               
               if jobCount == 0 
                  break
               end
               
               startP = stopP + 1
               stopP = min(stopP + b, nQ)
           end  
       else       
           @inbounds @views for j in workers()
               try
                     idxs_seg, dist_seg = fetch(popfirst!(placeholders))
               catch ex
                     println(ex)
               end
               
               jobCount -= 1
               
               nearestKN[1:k+1,startP:stopP] .= idxs_seg[1:k+1,:]
               distsKN[1:k+1,startP:stopP] .= dist_seg[1:k+1,:]
               
               if jobCount == 0 
                  break
               end
               
               startP = stopP + 1
               stopP = min(stopP + b, nQ)
           end
       end

      idxs_seg = nothing
      dist_seg = nothing
     
     return nearestKN, distsKN
end

function distributedKNN(
      filenameC::String,
      filenameQ::String,      
      file_indxs_out::String, 
      file_dists_out::String, 
      k::Int, 
      d::Int = 28, 
      nC::Int = 11000000,
      nQ::Int = 11000000,      
      cSize::Int = 100000, 
      qSize::Int = 11000,
      algorithm::Int = 3,    # Algorithm 0 == FLANN (algorithm 0)
                            # Algorithm 1 == FLANN (approximated)
                            # Algorithm 2 == GPU 
                            # Algorithm 3 == CPU
                            # Algorithm 4 == ParallelNeighbors
      in_memory::Bool = false
      )

      nw = nworkers()
      np = Threads.nthreads()      
      
      b = ceil(Int64, nQ/nw)
      
      if in_memory == true
          idxs_seg =  zeros(Int32,k+1,b)
          dist_seg = zeros(Float32,k+1,b)
      end
     
      placeholders = []
      
      numJobs = ceil(Int32, nQ/qSize)
      progress = RemoteChannel(()->Channel{Tuple}(numJobs));
      
      if isfile(file_indxs_out)
         rm(file_indxs_out)
      end
      if isfile(file_dists_out)
         rm(file_dists_out)
      end
      
      GPUs = Int(length(devices()))
      gpuID = 0
      
      println("Starting distributed kNN calculation of $nQ Query points from file \"$filenameQ\" and $nC Corpus points from file \"$filenameC\"")
      println(" Number of workers: $nw")
      println(" Number of threads: $np")
      if algorithm == 2 || algorithm == 4
         
         println(" Number of GPUs: $GPUs")
      end
      println(" Total number of job segments: $numJobs")
      
      jobCount = numJobs
      
      #@sync begin
      startP = 1
      stopP = b
      for j in workers()
         if in_memory == false
               if algorithm == 0
                     pl = @spawnat j distPointKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true)
                     jobCount -= 1
                elseif algorithm == 1 
                     pl = @spawnat j distPointKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, false)
                     jobCount -= 1
                elseif algorithm ==2
                     pl = @spawnat j distPointKNN_GPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     gpuID = (gpuID + 1) % GPUs 
                     jobCount -= 1
                elseif algorithm ==3
                     pl = @spawnat j distPointKNN_CPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k) 
                     jobCount -= 1
                else
                     pl = @spawnat j distPointParKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true, gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                end
         else
               if algorithm == 0
                     pl = @spawnat j distPointKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true)
                     jobCount -= 1
                elseif algorithm == 1 
                     pl = @spawnat j distPointKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, false)
                     jobCount -= 1
                elseif algorithm ==2
                     pl = @spawnat j distPointKNN_GPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     gpuID = (gpuID + 1) % GPUs 
                     jobCount -= 1
                elseif algorithm ==3
                     pl = @spawnat j distPointKNN_CPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k) 
                     jobCount -= 1
                else
                     pl = @spawnat j distPointParKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true, gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                end
         end
       
         push!(placeholders,pl)
         
        if jobCount == 0 
            break
        end
         
         startP = stopP + 1
         stopP = min(stopP + b, nQ)
      end
       #end
       
       jobCount = numJobs
       counter = 0
       if algorithm == 2 || algorithm == 4
           while jobCount > 0
              job_id, local_jobs, gpu_ID, myid = take!(progress)
              counter += 1
              println("  $counter of $numJobs done: Worker $myid finished local job $job_id of $local_jobs on GPU $gpu_ID")
              jobCount -= 1
           end
       else
           while jobCount > 0
              job_id, local_jobs, myid = take!(progress)
              counter += 1
              println("  $counter of $numJobs done: Worker $myid finished local job $job_id of $local_jobs")
              jobCount -= 1
           end
       end
       
       startP = 1
       stopP = b
       
       jobCount = numJobs
       counter = 0
       if in_memory == false
           @inbounds for j in workers()
               filename_indxs, filename_dists = fetch(popfirst!(placeholders))
               #try
               #    filename_indxs, filename_dists = fetch(popfirst!(placeholders))
               #catch ex
               #    println(ex)
               #end
               
               jobCount -= 1
           
               if counter == 0
                   store_bin_file(file_indxs_out, k+1, nQ, filename_indxs, false, Int32)
                   store_bin_file(file_dists_out, k+1, nQ, filename_dists, false, Float32)
               else
                   store_bin_file(file_indxs_out, k+1, nQ, filename_indxs, true, Int32)
                   store_bin_file(file_dists_out, k+1, nQ, filename_dists, true, Float32)
               end
               
               if isfile(filename_indxs)
                  rm(filename_indxs)
               end
               if isfile(filename_dists)
                  rm(filename_dists)
               end
               
               counter += 1
               
               if jobCount == 0 
                  break
               end
               
               startP = stopP + 1
               stopP = min(stopP + b, nQ)
           end  
       else       
           @inbounds @views for j in workers()
               try
                     idxs_seg, dist_seg = fetch(popfirst!(placeholders))
               catch ex
                     println(ex)
               end
               
               jobCount -= 1
               
               if counter == 0
                   store_file(file_indxs_out, idxs_seg, false)
                   store_file(file_dists_out, dist_seg, false)
               else
                   store_file(file_indxs_out, idxs_seg, true)
                   store_file(file_dists_out, dist_seg, true)
               end
               
               counter += 1
               
               if jobCount == 0 
                  break
               end
               
               startP = stopP + 1
               stopP = min(stopP + b, nQ)
           end
       end
       
       if in_memory == true
          idxs_seg =  nothing
          dist_seg = nothing
      end
end
