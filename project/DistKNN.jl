using FLANN, BenchmarkTools, Statistics, Distributed, DelimitedFiles, ParallelNeighbors, CUDA, Random, Clustering, StatsBase, LinearAlgebra, NearestNeighbors, Faiss

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
         
         di = [Pair(i, d) for (i,d) in enumerate(Distance[1:stopP])]
         
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
       error("Attempting to copy arrays with mismatching number of collumns: dest cols: $size(dest,2), source cols: $n")
    elseif size(dest,1) != d 
       error("Attempting to copy arrays with mismatching number of rows: dest rows: size(dest,1), source rows: $d ")
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
     
     pt1 = ones(Int64, np)
     pt2 = ones(Int64, np)
    
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
        println("Size mismatch for the merging results, idx1 $n, idx2 $(size(idx2, 2)), dist1 $(size(dist1, 2)), dist2 $(size(dist2, 2)), Indx, $(size(Indx,2)), Dists $(size(Dists, 2))")
        return
     end
     if k != size(idx2, 1) || k != size(dist1, 1) || k != size(dist2, 1) || k != size(Indx,1) || k != size(Dists, 1)
        println("Dimension mismatch for the merging results")
        return
     end

     np = Threads.nthreads()
     
     
     pt1 = ones(Int64, np)
     pt2 = ones(Int64, np)
    
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
            idxs_seg[1:k,l] .= partialsortperm!(tempIndex[1:c_length,l],distance[1:c_length,l],1:k,initialized=false) .+ cOff
            dist_seg[1:k,l] .= distance[tempIndex[1:k,l],l]
            
            nothing
       end 
       
       return       
end

function randmatrix(
     r::Int,
     d::Int
     )
     
     rnd = RandomDevice()
     
     R = zeros(Float32, r, d)
     
     sqp = 1.0f0 * sqrt(3)
     sqm = -1.0f0 * sqrt(3)
     
     num = 1
     
     @inbounds for i in 1:r
         for j in 1:d
             num = rand(rnd, 1:6)
             
             if num == 1
                 R[i,j] = sqp
             elseif num == 6
                 R[i,j] = sqm
             end
             
         end
     end
     
     return R
end

function randmatrix!(
     R::AbstractArray{Float32},
     r::Int,
     d::Int
     )
     
     rnd = RandomDevice()
     
     sqp = 1.0f0 * sqrt(3)
     sqm = -1.0f0 * sqrt(3)
     
     num = 1
     
     @inbounds for i in 1:r
         for j in 1:d
             num = rand(rnd, 1:6)
             
             if num == 1
                 R[i,j] = sqp
             elseif num == 6
                 R[i,j] = sqm
             end
             
         end
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
        
        
        
        if d > 3
            @inbounds for j in 1:nC
                lastj = d
                
                sum0 = 0.0f0
                sum1 = 0.0f0
                sum2 = 0.0f0
                sum3 = 0.0f0
                sum4 = 0.0f0
                @inbounds @simd for k in 1:4:d-3
                         
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
        
         @inbounds for i in 1:nQ
         
            for j in 1:nC
                tempDist = 0.0f0
                temp = 0.0f0

                for k in 1:d                   
                    temp = Q[k,i] - C[k,j]
                    tempDist += temp*temp
                end

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
                 
                 @inbounds for j in 1:d
                   tempDist += (Q_d[j,yIndex] - C_d[j,xIndex])^2
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
                     NQ = stopQ
                  end
                  if stopC < nC
                     NC = stopC
                  end
               end
               
               kernel = @cuda name="euclDistGPUb" launch=false euclDistGPUb(Q_d, C_d, Distance_d, NQ, NC, d)
               config = launch_configuration(kernel.fun)
               threads = Base.min(NC*NQ, config.threads)
               
               blockSize = floor(Int,sqrt(threads))
               sqBlock = isqrt(blockSize)
               sblX = min(NC, sqBlock)
               sblY = min(NQ, sqBlock)

               nblX = ceil(Int32, NC/sblX)
               nblY = ceil(Int32, NQ/sblY)
               
               blockSizeRect = (sblX, sblY)
               nblRect = (nblX, nblY)
               
               kernel(Q_d, C_d, Distance_d, NQ, NC, d; threads=blockSizeRect, blocks=nblRect)
                  
               return
end

@inbounds function euclDistGPUb_filter(
            Q_d, 
            C_d, 
            Distance_d,
            candidateList_d,            
            nQ::Int, 
            nC::Int, 
            d::Int
            )
		    xIndex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            yIndex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
            
            
            
            if (xIndex <= nC) && (yIndex <= nQ)
                 if(@inbounds candidateList_d[xIndex,yIndex] == 0)
                    @inbounds Distance_d[xIndex,yIndex] = typemax(Float32)
                    return
                 end
                 
                 tempDist = 0.0f0
                 
                 @inbounds for j in 1:d
                   tempDist += (Q_d[j,yIndex] - C_d[j,xIndex])^2
                 end
                 @inbounds Distance_d[xIndex,yIndex] = tempDist
			end
            return
end


@inline function euclDistGPU_filter_wrapper!(
               Q_d, 
               C_d, 
               Distance_d,
               candidateList_d,               
               nQ::Int, 
               nC::Int, 
               d::Int,
               stopQ::Int,
               stopC::Int,
               process::Int
               )
               
               NQ = nQ
               NC = nC
               
               if process != 1
                  if stopQ < nQ
                     NQ = stopQ
                  end
                  if stopC < nC
                     NC = stopC
                  end
               end
               
               kernel = @cuda name="euclDistGPU_filter_wrapper" launch=false euclDistGPUb_filter(Q_d, C_d, Distance_d, candidateList_d, NQ, NC, d)
               config = launch_configuration(kernel.fun)
               threads = Base.min(NC*NQ, config.threads)
               
               blockSize = floor(Int,sqrt(threads))
               sqBlock = isqrt(blockSize)
               sblX = min(NC, sqBlock)
               sblY = min(NQ, sqBlock)
               
               nblX = ceil(Int32, NC/sblX)
               nblY = ceil(Int32, NQ/sblY)
               
               blockSizeRect = (sblX, sblY)
               nblRect = (nblX, nblY)
               
               kernel(Q_d, C_d, Distance_d, candidateList_d, NQ, NC, d; threads=blockSizeRect, blocks=nblRect)
                
               return
end

@inline function euclDistFromCen(
        Q::AbstractArray{Float32},
        C::AbstractArray{Float32},
        Distance::AbstractArray{Float32},
        assignments::Vector{Int32},
        pointQ::Int, 
        nC::Int, 
        d::Int
        )
        
        @inbounds for j in 1:nC
            if(assignments[j]!=pointQ)
               Distance[j,pointQ] = typemax(Float32)
               
               continue
            end
            
            tempDist = 0.0f0

            @inbounds for k in 1:d
                tempDist += (Q[k,pointQ] - C[k,j])^2
            end

            @inbounds Distance[j,pointQ] = tempDist
            end
        return
end

@inline function euclDistFromCenGPU_wrapper!(
               Q_d, 
               C_d, 
               Distance_d,
               assignments_d,               
               nQ::Int, 
               nC::Int, 
               d::Int,
               stopQ::Int,
               stopC::Int,
               process::Int
               )
               
               NQ = nQ
               NC = nC
               
               if process != 1
                  if stopQ < nQ
                     NQ = stopQ
                  end
                  if stopC < nC
                     NC = stopC
                  end
               end
               
               kernel = @cuda name="euclDistFromCenGPU_wrapper" launch=false euclDistFromCenGPU(Q_d, C_d, Distance_d, assignments_d, NQ, NC, d)
               config = launch_configuration(kernel.fun)
               threads = Base.min(NC*NQ, config.threads)
               
               blockSize = floor(Int,sqrt(threads))
               sqBlock = isqrt(blockSize)
               sblX = min(NC, sqBlock)
               sblY = min(NQ, sqBlock)
               
               nblX = ceil(Int32, NC/sblX)
               nblY = ceil(Int32, NQ/sblY)
               
               blockSizeRect = (sblX, sblY)
               nblRect = (nblX, nblY)
               
               kernel(Q_d, C_d, Distance_d, assignments_d, NQ, NC, d; threads=blockSizeRect, blocks=nblRect)
                
               return
end

@inline function euclDistFromCen(
        Q::AbstractArray{Float32},
        C::AbstractArray{Float32},
        Distance::AbstractArray{Float32},
        assignments::Vector{Int32},
        pointQ::Int, 
        nC::Int, 
        d::Int
        )
        
        @inbounds for j in 1:nC
            if(assignments[j]!=pointQ)
               Distance[j,pointQ] = typemax(Float32)
               
               continue
            end
            
            tempDist = 0.0f0

            @inbounds for k in 1:d
                tempDist += (Q[k,pointQ] - C[k,j])^2
            end

            @inbounds Distance[j,pointQ] = tempDist
            end
        return
end

function cenTI_filter(
             C::AbstractArray,
             Q::AbstractArray,
             d::Int,
             k::Int
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             clSizeC = min(3 * Int(ceil(sqrt(cSize))),cSize)
             clSizeQ = min(3 * Int(ceil(sqrt(qSize))),qSize)
             
             
             aQ = zeros(Int32, qSize)
             MQ = zeros(Float32, d, clSizeQ)
             
             clstQ = kmeans(Q, clSizeQ; maxiter=3, display=:none)
             vQ = assignments(clstQ)
             VMQ = clstQ.centers
             
             @inbounds for i in 1:qSize
                 aQ[i] = vQ[i]
             end
             
             @inbounds for j in 1:clSizeQ
                 for i in 1:d
                    MQ[i,j] = VMQ[i,j]
                 end
             end
             
             aC = zeros(Int32, cSize)
             MC = zeros(Float32, d, clSizeC)
             
             clstC = kmeans(C, clSizeC; maxiter=3, display=:none)
             vC = assignments(clstC)
             VMC = clstC.centers
             
             @inbounds for i in 1:cSize
                 aC[i] = vC[i]
             end
             
             @inbounds for j in 1:clSizeC
                 for i in 1:d
                    MC[i,j] = VMC[i,j]
                 end
             end
             
             vQ = nothing
             VMQ = nothing
             
             vC = nothing
             VMC = nothing
             
             clstQ = nothing
             clSizeC = nothing

             return TI_filter(C,Q,d,k,MC,aC,MQ,aQ)
end

function cenTI_filter(
             C::AbstractArray,
             Q::AbstractArray,
             d::Int,
             k::Int,
             MQ,
             aQ
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             
             MC, aC = cenTI_cluster(C, d, k)
             
             return TI_filter(C,Q,d,k,MC,aC,MQ,aQ)
end

function cenTI_cluster(
             Q::AbstractArray,
             d::Int,
             k::Int
             )
             
             qSize = size(Q,2)
             
             clSizeQ = min(3 * Int(ceil(sqrt(qSize))),qSize)
             
             aQ = zeros(Int32, qSize)
             MQ = zeros(Float32, d, clSizeQ)
             
             clstQ = kmeans(Q, clSizeQ; maxiter=3, display=:none)
             vQ = assignments(clstQ)
             VMQ = clstQ.centers
             
             @inbounds for i in 1:qSize
                 aQ[i] = vQ[i]
             end
             
             @inbounds for j in 1:clSizeQ
                 for i in 1:d
                    MQ[i,j] = VMQ[i,j]
                 end
             end
             
             vQ = nothing
             VMQ = nothing
             
             clstQ = nothing

             return (MQ, aQ)
end

function TI_filter(
             C::AbstractArray,
             Q::AbstractArray,
             d::Int,
             k::Int,
             MC,
             aC,
             MQ,
             aQ
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             clSizeC = size(MC,2)
             clSizeQ = size(MQ,2)
             
             checkList = zeros(Int32, cSize,qSize)
             
             idxs_cen = zeros(Int32, clSizeC, clSizeQ)
             dist_cen = zeros(Float32, clSizeC, clSizeQ)
             distance_cen = zeros(Float32, clSizeC, clSizeQ)
             tIndx_cen = zeros(Int32,clSizeC, clSizeQ)
             
             maxDist_Q = zeros(Float32, 1, clSizeQ)
             distfromCen_Q = zeros(Float32, qSize, clSizeQ)
             
             maxDist_C = zeros(Float32, 1, clSizeC)
             distance_c = zeros(Float32, cSize, clSizeC)
             distfromCen_C = zeros(Float32, cSize, clSizeC)
             idxsfromCen_C = zeros(Int32, cSize, clSizeC)
             tIndxfromCen_C = zeros(Int32,cSize, clSizeC)
             
             
             @sync begin
                 #Calculate distances between cluster centers for Q and C
                 @inbounds @views Threads.@threads for l in 1:clSizeQ
                     euclDistP(MQ,MC,distance_cen,l,clSizeC,d)                   
                     idxs_cen[1:clSizeC,l] .= sortperm!(tIndx_cen[1:clSizeC,l],distance_cen[1:clSizeC,l],initialized=false)
                     dist_cen[1:clSizeC,l] .= distance_cen[idxs_cen[1:clSizeC,l],l]
                        
                     nothing
                 end
             end
             
             idxs_cen = nothing
             distance_cen = nothing
             tIndx_cen = nothing
             
             @sync begin
                 #Calculate distance of points of Q from their cluster centers and keep the max too
                 @inbounds @views Threads.@threads for l in 1:clSizeQ
                     euclDistFromCen(MQ,Q,distfromCen_Q,aQ,l,qSize,d)
                     maxDist_Q[l] = maximum(distfromCen_Q[1:qSize,l])
                        
                     nothing
                 end
             end
             
             distfromCen_Q = nothing
             
             @sync begin
                 #Calculate distance of points of C from their cluster centers
                 @inbounds @views Threads.@threads for l in 1:clSizeC
                     euclDistFromCen(MC,C,distance_c,aC,l,cSize,d)                   
                     idxsfromCen_C[1:cSize,l] .= sortperm!(tIndxfromCen_C[1:cSize,l],distance_c[1:cSize,l],rev = true,initialized=false)
                     distfromCen_C[1:cSize,l] .= distance_c[idxsfromCen_C[1:cSize,l],l]
                     maxDist_C[l] = distfromCen_C[1,l]
                        
                     nothing
                 end
             end
             
             distance_c = nothing
             tIndxfromCen_C = nothing
             idxsfromCen_C = nothing
             MQ = nothing
             MC = nothing
             
             kmax = min(k, cSize)
             candidateList = zeros(Int32, clSizeC, clSizeQ)
             upperBounds = zeros(Float32, kmax, clSizeC, clSizeQ)
             UBvec = zeros(Float32, kmax * clSizeC, clSizeQ)
             qUB = zeros(Float32, 1, clSizeQ)
             
             @sync begin
                 #calculate upper bounds for all query clusters
                 @inbounds @views Threads.@threads for l in 1:clSizeQ
                    for j in 1:clSizeC
                       for i in 1:kmax
                           upperBounds[i,j,l] = maxDist_Q[l] + dist_cen[j,l] + distfromCen_C[max(cSize - i + 1,1),j]
                       end
     
                    end
                    UBvec[1:kmax * clSizeC,l] .= reshape(upperBounds[1:kmax,1:clSizeC,l], kmax * clSizeC)
                   sort(UBvec[1:kmax * clSizeC,l])
                   qUB[l] = UBvec[kmax,l]
                 
                    nothing
                 end
             end
             
             
             upperBounds = nothing
             UBvec = nothing
             
             lowerBounds = zeros(Float32, clSizeC, clSizeQ)
             clusterList = ones(Int32, clSizeC, clSizeQ)
             candidateList = zeros(Int32, cSize, qSize)
             
             @sync begin
                 #create the filter list of target clusters too far from each query clusters
                 # if 0 it's too far              
                 @inbounds Threads.@threads for l in 1:clSizeQ
                    for j in 1:clSizeC
                        lowerBounds[j,l] = dist_cen[j,l] - maxDist_Q[l] - maxDist_C[j]
                        if( qUB[l] < lowerBounds[j,l])
                            clusterList[j,l] = 0
                        end
                        
                    end
                 
                    nothing
                 end
             end
             
             
             maxDist_Q = nothing
             maxDist_C = nothing
             
             @sync begin
                 @inbounds Threads.@threads for w in 1:qSize
                    for z in 1:cSize
                        if ((clusterList[aC[z],aQ[w]] !=0) && (abs(dist_cen[aC[z],aQ[w]] - distfromCen_C[z,aC[z]]) <= qUB[aQ[w]]) )
                           candidateList[z,w] += 1
                        end

                    end
                    
                    nothing
                end
            end
            
           @sync begin
                checkList = nothing
                
                
                dist_cen = nothing
        
                distfromCen_C = nothing
                
                
                clstQ = nothing
                aQ = nothing
                
                clstC = nothing
                aC = nothing
                
              
                qUB = nothing
                lowerBounds = nothing
                clusterList = nothing
            end
            
            return candidateList
end

function cenTI_filter_GPU(
             C::AbstractArray,
             Q::AbstractArray,
             d::Int,
             k::Int
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             clSizeC = 3 * Int(ceil(sqrt(cSize)))
             clSizeQ = 3 * Int(ceil(sqrt(qSize)))
             
             clstQ = kmeans(Q, clSizeQ; maxiter=3, display=:none)
             aQ = assignments(clstQ)
             MQ = clstQ.centers
             

             clstC = kmeans(C, clSizeC; maxiter=3, display=:none)
             aC = assignments(clstC)
             MC = clstC.centers

             return TI_filter(C,Q,d,k,MC,aC,MQ,aQ)
end

function cenTI_filter_GPU(
             C::AbstractArray,
             Q::AbstractArray,
             d::Int,
             k::Int,
             MQ,
             aQ
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             
             MC, ac = cenTI_cluster(C, d, k)
             
             return TI_filter(C,Q,d,k,MC,aC,MQ,aQ)
end


@inbounds function euclDist_filter(
        Q,
        C,
        Distance,
        candidateList,
        nQ::Int, 
        nC::Int, 
        d::Int
        )
        
         @inbounds Threads.@threads for i in 1:nQ
         
            @inbounds for j in 1:nC
                
                if candidateList[j,i] == 0
                    Distance[j,i] = typemax(Float32)
                    continue
                end
                
                tempDist = 0.0f0
                temp = 0.0f0

                @inbounds for k in 1:d
                    
                    temp = Q[k,i] - C[k,j]
                    tempDist += temp*temp
                end
                Distance[j,i] = tempDist
            end
            
            nothing
        end
        
        return
end

function rpKNN_CPU(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             r::Int,
             P::Int
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
             
             
             rpKNN_CPU!(Q,C,d,k,r,P,idxs,dist)
             
             return idxs, dist
end

function rpKNN_CPU!(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             r::Int,
             P::Int,
             idxs::AbstractArray,
             dist::AbstractArray
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             #r = Int(ceil(sqrt(d)))
             
             if P <= 0
                  P = 3
             end
             
             if size(idxs,2) < qSize || size(dist,2) < qSize
                error("The output matrixes of rpKNN_CPU! are smaller than the input matrixes")
             end
             
             kr = max(4*k, Int(ceil(cSize/10)))
             kr = min(kr, cSize-1)
             
             idxs_rp = zeros(Int32, kr, qSize)
             dist_rp = zeros(Float32, kr, qSize)
             
             distance = zeros(Float32,cSize,qSize)
             tempIndex = zeros(Int32,cSize,qSize)
             
             R = zeros(Float32, r, d)
             
             Cr = zeros(Float32, r, cSize)
             Qr = zeros(Float32, r, qSize)
             
             candidateList = zeros(Int32, cSize, qSize)
             
             @views @inbounds for rp in 1:P
                 randmatrix!(R,r,d)
                            
                 mul!(Cr,R,C)
                 mul!(Qr,R,Q)
                 
                 @views @inbounds Threads.@threads for l in 1:qSize
                         euclDistP(Qr,Cr,distance,l,cSize,r)                   
                         idxs_rp[1:kr,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:kr,initialized=false)
                         dist_rp[1:kr,l] .= distance[tempIndex[1:kr,l],l]
                         
                         for j in 1:kr
                             candidateList[idxs_rp[j,l],l] += 1
                         end
                            
                         nothing
                 end  
                 if rp%5 == 0 
                     GC.gc()
                 end
             end
             
             euclDist_filter(Q,C,distance,candidateList,qSize,cSize,d)
             
             @views @inbounds Threads.@threads for l in 1:qSize
                    idxs[1:k,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:k,initialized=false)
                    dist[1:k,l] .= distance[tempIndex[1:k,l],l]
                    
                    nothing
            end
             
            return
end

function rpKNN_GPU(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             r::Int,
             P::Int
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
             
             
             rpKNN_GPU!(Q,C,d,k,r,P,idxs,dist)
             
             return idxs, dist
end


function rpKNN_GPU!(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             r::Int,
             P::Int,
             idxs::AbstractArray,
             dist::AbstractArray
             )
             
             qSize = size(Q,2)
             cSize = size(C,2)           
             
             if P <= 0
                  P = 3
             end
             
             if size(idxs,2) < qSize || size(dist,2) < qSize
                error("The output matrixes of rpKNN_GPU! are smaller than the input matrixes")
             end
             
             C_d = CuArray(C)
             Q_d = CuArray(Q)
             
             
             kr = max(4*k, Int(ceil(cSize/10)))
             kr = min(kr, cSize-1)
             
             idxs_rp = zeros(Int32, kr, qSize)
             dist_rp = zeros(Float32, kr, qSize)
             
             distance = CUDA.Mem.pin(zeros(Float32,cSize,qSize))
             tempIndex = zeros(Int32,cSize,qSize)
             
             distance_d_even = CuArray(distance)
             distance_d_odd = CuArray(distance)
             
             R = CUDA.Mem.pin(zeros(Float32, r, d))
             
             R_d = CuArray(R)
             
             Cr_d = CuArray(zeros(Float32, r, cSize))
             Qr_d = CuArray(zeros(Float32, r, qSize))
             
             candidateList = zeros(Int32, cSize, qSize)
             candidateList_d = CuArray(candidateList)
             
             
             sort_finished = @async nothing
             
             @views @inbounds for rp in 1:P
                 randmatrix!(R,r,d)
                 
                 copyto!(R_d, R) 
                 
                 CUDA.@sync mul!(Cr_d,R_d,C_d)
                 CUDA.@sync mul!(Qr_d,R_d,Q_d)
                 
                 if iseven(rp)
                        CUDA.@sync euclDistGPU_wrapper!(Qr_d,Cr_d,distance_d_even,qSize,cSize,r,length(1:qSize),length(1:cSize),2)                     
                 else
                        CUDA.@sync euclDistGPU_wrapper!(Qr_d,Cr_d,distance_d_odd,qSize,cSize,r,length(1:qSize),length(1:cSize),2)
                 end
                 
                 wait(sort_finished)
                 
                 distDone = iseven(rp) ? 0 : 1
                 
                 sort_finished = @async begin
                 
                    if distDone == 0
                        copyto!(distance, distance_d_even)
                    else
                        copyto!(distance, distance_d_odd)
                    end
                 
                     
                     @inbounds @views Threads.@threads for l in 1:qSize   
                             idxs_rp[1:kr,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:kr,initialized=false)
                             dist_rp[1:kr,l] .= distance[tempIndex[1:kr,l],l]
                             
                             for j in 1:kr
                                 candidateList[idxs_rp[j,l],l] += 1
                             end
                             
                             nothing
                     end
                 end 
                 if rp%5 == 0 
                     GC.gc()
                 end                 
             end
             
             wait(sort_finished)
             
             copyto!(candidateList_d, candidateList)
             euclDistGPU_filter_wrapper!(Q_d,C_d,distance_d_even,candidateList_d,qSize,cSize,d,length(1:qSize),length(1:cSize),2)
             
             copyto!(distance, distance_d_even)
             @views @inbounds Threads.@threads for l in 1:qSize
                    idxs[1:k,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:k,initialized=false)
                    dist[1:k,l] .= distance[tempIndex[1:k,l],l]
                    
                    nothing
            end
            
            candidateList = nothing
            distance = nothing
            tempIndex = nothing
            R = nothing
            idxs_rp = nothing
            dist_rp = nothing
            
            CUDA.unsafe_free!(Q_d)
            CUDA.unsafe_free!(C_d)
            CUDA.unsafe_free!(Qr_d)
            CUDA.unsafe_free!(Cr_d)
            CUDA.unsafe_free!(R_d)
            CUDA.unsafe_free!(distance_d_even)
            CUDA.unsafe_free!(distance_d_odd)
            CUDA.unsafe_free!(candidateList_d)
             
            return
end


function cenKNN_GPU(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
             
             
             cenKNN_GPU!(Q,C,d,k,idxs,dist)
             
             return idxs, dist
end

function cenKNN_GPU!(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             idxs::AbstractArray,
             dist::AbstractArray
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             if size(idxs,2) < qSize || size(dist,2) < qSize
                error("The output matrixes of rpKNN_GPU! are smaller than the input matrixes")
             end
             
             C_d = CuArray(C)
             Q_d = CuArray(Q)

             
             distance = CUDA.Mem.pin(zeros(Float32,cSize,qSize))
             tempIndex = zeros(Int32,cSize,qSize)
             
             distance_d = CuArray(distance)
             
             
             candidateList = zeros(Int32, cSize, qSize)
             candidateList_d = CuArray(candidateList)
             
             candidateList .= cenTI_filter(C,Q,d,k)
             
             copyto!(candidateList_d, candidateList)
             euclDistGPU_filter_wrapper!(Q_d,C_d,distance_d,candidateList_d,qSize,cSize,d,length(1:qSize),length(1:cSize),2)
             
             copyto!(distance, distance_d)
             @views @inbounds Threads.@threads for l in 1:qSize
                    idxs[1:k,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:k,initialized=false)
                    dist[1:k,l] .= distance[tempIndex[1:k,l],l]
                    
                    nothing
            end
            
            distance = nothing
            tempIndex = nothing
            candidateList = nothing
            
            CUDA.unsafe_free!(Q_d)
            CUDA.unsafe_free!(C_d)
            CUDA.unsafe_free!(distance_d)
            CUDA.unsafe_free!(candidateList_d)
             
            return
end

function cenKNN_GPU!(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             idxs::AbstractArray,
             dist::AbstractArray,
             MQ,
             aQ
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             if size(idxs,2) < qSize || size(dist,2) < qSize
                error("The output matrixes of rpKNN_GPU! are smaller than the input matrixes")
             end
             
             C_d = CuArray(C)
             Q_d = CuArray(Q)

             
             distance = CUDA.Mem.pin(zeros(Float32,cSize,qSize))
             tempIndex = zeros(Int32,cSize,qSize)
             
             distance_d = CuArray(distance)
             
             
             candidateList = zeros(Int32, cSize, qSize)
             candidateList_d = CuArray(candidateList)
             
             candidateList .= cenTI_filter(C,Q,d,k,MQ,aQ)
             
             copyto!(candidateList_d, candidateList)
             euclDistGPU_filter_wrapper!(Q_d,C_d,distance_d,candidateList_d,qSize,cSize,d,length(1:qSize),length(1:cSize),2)
             
             copyto!(distance, distance_d)
             @views @inbounds Threads.@threads for l in 1:qSize
                    idxs[1:k,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:k,initialized=false)
                    dist[1:k,l] .= distance[tempIndex[1:k,l],l]
                    
                    nothing
            end
            
            distance = nothing
            tempIndex = nothing
            candidateList = nothing
            
            CUDA.unsafe_free!(Q_d)
            CUDA.unsafe_free!(C_d)
            CUDA.unsafe_free!(distance_d)
            CUDA.unsafe_free!(candidateList_d)
             
            return
end

function cenKNN_CPU(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
             
             
             cenKNN_CPU!(Q,C,d,k,idxs,dist)
             
             return idxs, dist
end

function cenKNN_CPU!(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             idxs::AbstractArray,
             dist::AbstractArray
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             distance = zeros(Float32,cSize,qSize)
             tempIndex = zeros(Int32,cSize,qSize)            
             
             candidateList = zeros(Int32, cSize, qSize)
             
             candidateList .= cenTI_filter(C,Q,d,k)
             
             euclDist_filter(Q,C,distance,candidateList,qSize,cSize,d)
             
             candidateList = nothing
             GC.gc(true)
             
             @views @inbounds Threads.@threads for l in 1:qSize
                    idxs[1:k,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:k,initialized=false)
                    dist[1:k,l] .= distance[tempIndex[1:k,l],l]
                    
                    nothing
            end
            
            distance = nothing
            tempIndex = nothing
            
             
            return
end

function cenKNN_CPU!(
             Q::AbstractArray,
             C::AbstractArray,
             d::Int,
             k::Int,
             idxs::AbstractArray,
             dist::AbstractArray,
             MQ,
             aQ
             )
             
             cSize = size(C,2)
             qSize = size(Q,2)
             
             distance = zeros(Float32,cSize,qSize)
             tempIndex = zeros(Int32,cSize,qSize)            
             
             candidateList = zeros(Int32, cSize, qSize)
             
             candidateList .= cenTI_filter(C,Q,d,k,MQ,aQ)
             
             euclDist_filter(Q,C,distance,candidateList,qSize,cSize,d)
             
             candidateList = nothing
             GC.gc(true)
             
             @views @inbounds Threads.@threads for l in 1:qSize
                    idxs[1:k,l] .= partialsortperm!(tempIndex[1:cSize,l],distance[1:cSize,l],1:k,initialized=false)
                    dist[1:k,l] .= distance[tempIndex[1:k,l],l]
                    
                    nothing
            end
            
            distance = nothing
            tempIndex = nothing
             
            return
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
             
               
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_vec = [Vector{Int32}(undef, k) for _ in 1:qSize]
             dists_vec = [Vector{Float32}(undef, k) for _ in 1:qSize]
    
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)

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
                   
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)
            
                   if hybrid_all == true
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], kmax; algorithm=:hybrid_batch_all)                     
                   else
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], kmax; algorithm=:hybrid_batch_test) 
                   end
                   
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:kmax
                             idxs_seg[jj,ii] = idxs_vec[ii][jj] + cOff
                             dist_seg[jj,ii] = dists_vec[ii][jj]
                         end
                   
                   end
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:length(qStart:qStop)] .= typemax(Float32)
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
             
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = similar(idxs)
             dist_seg = similar(dist)
             
             idxs_vec = [Vector{Int32}(undef, k) for _ in 1:qSize]
             dists_vec = [Vector{Float32}(undef, k) for _ in 1:qSize]
             
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
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
                   
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)
                   
                   if hybrid_all == true
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], kmax; algorithm=:hybrid_batch_all)                     
                   else
                       idxs_vec, dists_vec = ParallelNeighbors.knn(C[:,1:length(cStart:cStop)], Q[:,1:length(qStart:qStop)], kmax; algorithm=:hybrid_batch_test) 
                   end
                   
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:kmax
                             idxs_seg[jj,ii] = idxs_vec[ii][jj] + cOff
                             dist_seg[jj,ii] = dists_vec[ii][jj]
                         end
                   
                   end
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:length(qStart:qStop)] .= typemax(Float32)
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
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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

function distPointFaiss_fileout(
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
             
               
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
    
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)

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
                   
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)
            
                   idx = Index(d; str="IDMap2,Flat", metric="L2", gpus="$gpuID")
                   
                   vs_gallery = C[:,1:length(cStart:cStop)]';
                   vs_query = Q[:,1:length(qStart:qStop)]';
                   
                   ids = collect(range(1; stop = c_length))
                   
                   add_with_ids(idx, vs_gallery, ids)
                   dists_f, idxs_f = search(idx, vs_query, kmax) 
                   
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:kmax
                             idxs_seg[jj,ii] = idxs_f[ii,jj] + cOff -1
                             dist_seg[jj,ii] = dists_f[ii,jj]
                         end
                   
                   end
                   
                   idx = nothing
                   vs_gallery = nothing
                   vs_query = nothing
                   ids = nothing
                   dists_f = nothing
                   idxs_f = nothing
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:length(qStart:qStop)] .= typemax(Float32)
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
                   
                   GC.gc()
                   CUDA.reclaim()
                   
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
               CUDA.reclaim()
            end
      
      return (filename_indxs , filename_dists)
end

function distPointFaiss(
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
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)
             
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = similar(idxs)
             dist_seg = similar(dist)
             
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
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
                   
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)
                   
                   
                   idx = Index(d; str="IDMap2,Flat", metric="L2", gpus="$gpuID")
                   
                   vs_gallery = C[:,1:length(cStart:cStop)]';
                   vs_query = Q[:,1:length(qStart:qStop)]';
                   
                   ids = collect(range(1; stop = c_length))
                   
                   add_with_ids(idx, vs_gallery, ids)
                   dists_f, idxs_f = search(idx, vs_query, kmax) 
                   
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:kmax
                             idxs_seg[jj,ii] = idxs_f[ii,jj] + cOff -1 
                             dist_seg[jj,ii] = dists_f[ii,jj]
                         end
                   
                   end
                   
                   idx = nothing
                   vs_gallery = nothing
                   vs_query = nothing
                   ids = nothing
                   dists_f = nothing
                   idxs_f = nothing
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:length(qStart:qStop)] .= typemax(Float32)
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
                   
                   GC.gc()
                   CUDA.reclaim()
                   
                end
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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
               CUDA.reclaim()
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
             
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
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
                cStop = min(cSize, nC)
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
                   
                   kmax = min(k, c_length)
                   
                   merge_needed = j > 1 ? true : false
                   
                   distDone = iseven(j) ? 0 : 1 
                   
                   sort_finished = @async begin
                       if distDone == 0
                            copyto!(distance, distance_d_even)
                       else
                            copyto!(distance, distance_d_odd)
                       end
                       
                       calc_results!(idxs_seg, dist_seg, tempIndex, distance, c_length, q_length, kmax, cOff)
                       
                       
                       if kmax < k
                            dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
                       end
                
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
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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

         idxs = zeros(Int32, k, qSize)
         dist = zeros(Float32, k, qSize)
  
         idxs_seg = zeros(Int32, k, qSize)
         dist_seg = zeros(Float32, k, qSize)
         
         Indx = zeros(Int32, k, qSize)
         Dists = zeros(Float32, k, qSize)
         
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
         qStop = startP + qSize - 1
         qLocStart = 1
         qLocStop = qSize
         cOff = 0      
         
         @inbounds for i in 1:qBlock  
            load_file_wrapper!(filenameQ,d,Q, qStart:qStop)
            copyto!(Q_d, Q)
            
            cStart = 1
            cStop = min(cSize, nC)
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

               kmax = min(k, c_length)               
               
               merge_needed = j > 1 ? true : false
               
               distDone = iseven(j) ? 0 : 1
               
               
               sort_finished = @async begin
                   if distDone == 0
                        copyto!(distance, distance_d_even)
                   else
                        copyto!(distance, distance_d_odd)
                   end
                   
                   calc_results!(idxs_seg, dist_seg, tempIndex, distance, c_length, q_length, kmax, cOff)

                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
                   end
            
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


function distPointKNN_rp_GPU(
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
             r::Int,
             P::Int,
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
             
             
             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
     
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             distance = zeros(Float32,cSize,qSize)
             
             tempIndex = zeros(Int32,cSize,qSize)
             
             
             np = Threads.nthreads()
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
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)

                   cOff = cStart - 1 
                   
                   q_length = min(qSize, length(qStart:qStop)) 
                   c_length = min(cSize, length(cStart:cStop))

                   kmax = min(k, c_length)  
                   
                   rpKNN_GPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,r,P,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
                   idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
                   end
                   
                   merge_needed = j > 1 ? true : false
                   if merge_needed
                      @views mergeSeg!(idxs, dist, idxs_seg, dist_seg, Indx, Dists)
                   else
                      @views copy2dArr!(idxs, idxs_seg)
                      @views copy2dArr!(dist, dist_seg)
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
                         for w in 1:k
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
            tmp_idxs = nothing
            temp_dists = nothing            
      
      return (idxs_out , dist_out)
end

function distPointKNN_rp_GPU_fileout(
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
             r::Int,
             P::Int,
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
         
         cBlock = ceil(Int32, nC/cSize)
         qBlock = ceil(Int32, b/qSize)
          
         C = zeros(Float32, d, cSize)
         Q = zeros(Float32, d, qSize)
           
         idxs = zeros(Int32, k, qSize)
         dist = zeros(Float32, k, qSize)
         
         idxs_seg = zeros(Int32, k, qSize)
         dist_seg = zeros(Float32, k, qSize)
  
         Indx = zeros(Int32, k, qSize)
         Dists = zeros(Float32, k, qSize)
         
         distance = zeros(Float32,cSize,qSize)
         
         tempIndex = zeros(Int32,cSize,qSize) 
         
         np = Threads.nthreads()
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
            
            @inbounds @views for j in 1:cBlock
               load_file_wrapper!(filenameC,d,C, cStart:cStop)
               
               cOff = cStart - 1 
               
               q_length = min(qSize, length(qStart:qStop)) 
               c_length = min(cSize, length(cStart:cStop))  

               kmax = min(k, c_length)

               rpKNN_GPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,r,P,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
               idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff

               if kmax < k
                  dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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


function distPointKNN_cen_GPU(
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
             

             cBlock = ceil(Int32, nC/cSize)
             qBlock = ceil(Int32, b/qSize)
              
             C = zeros(Float32, d, cSize)
             Q = zeros(Float32, d, qSize)
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
     
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             distance = zeros(Float32,cSize,qSize)
             
             tempIndex = zeros(Int32,cSize,qSize)
             
             
             np = Threads.nthreads()
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
           
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)

                   cOff = cStart - 1 
                   
                   q_length = min(qSize, length(qStart:qStop))
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)
                   
                   cenKNN_GPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
                   idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
                   
                   GC.gc()
            
                   
                end
                
                MQ = nothing 
                aQ = nothing
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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

function distPointKNN_cen_GPU_fileout(
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
         
         cBlock = ceil(Int32, nC/cSize)
         qBlock = ceil(Int32, b/qSize)
          
         C = zeros(Float32, d, cSize)
         Q = zeros(Float32, d, qSize)
           
         idxs = zeros(Int32, k, qSize)
         dist = zeros(Float32, k, qSize)
         
         idxs_seg = zeros(Int32, k, qSize)
         dist_seg = zeros(Float32, k, qSize)
  
         Indx = zeros(Int32, k, qSize)
         Dists = zeros(Float32, k, qSize)
         
         distance = zeros(Float32,cSize,qSize)
         
         tempIndex = zeros(Int32,cSize,qSize) 
         
         np = Threads.nthreads()
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
            
            @inbounds @views for j in 1:cBlock
               load_file_wrapper!(filenameC,d,C, cStart:cStop)

               cOff = cStart - 1 
               
               q_length = min(qSize, length(qStart:qStop)) 
               c_length = min(cSize, length(cStart:cStop))
               
               kmax = min(k, c_length)               

               cenKNN_GPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
               idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff
               
               if kmax < k
                 dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
               
               GC.gc()
        
               
            end
            
            aQ = nothing
            MQ = nothing
           
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
               
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
            
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
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1

                   q_length = min(qSize, length(qStart:qStop))
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)                   
            
                   if bf == true
                       knn!(C[:,1:c_length], Q[:,1:q_length], kmax, FLANNParameters(algorithm=0), idxs_seg[1:kmax,1:q_length], dist_seg[1:kmax,1:q_length]);
                   else
                       knn!(C[:,1:c_length], Q[:,1:q_length], kmax, FLANNParameters(algorithm=4, trees=16,leaf_max_size=64,branching=32,iterations=14,centers_init=0,cb_index=0.2), idxs_seg[1:kmax,1:q_length], dist_seg[1:kmax,1:q_length]);
                   end
           
            
                   addOffset!(idxs_seg[1:kmax,1:q_length], 1, q_length, cOff)
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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

             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = similar(idxs)
             dist_seg = similar(dist)
             
             
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
            
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
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1

                   q_length = min(qSize, length(qStart:qStop))
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)                   
                   
                   if bf == true
                       knn!(C[:,1:c_length], Q[:,1:q_length], kmax, FLANNParameters(algorithm=0), idxs_seg[1:kmax,1:q_length], dist_seg[1:kmax,1:q_length]);
                   else
                       knn!(C[:,1:c_length], Q[:,1:q_length], kmax, FLANNParameters(algorithm=4, trees=16,leaf_max_size=64,branching=32,iterations=14,centers_init=0,cb_index=0.2), idxs_seg[1:kmax,1:q_length], dist_seg[1:kmax,1:q_length]);
                   end
           
            
                   addOffset!(idxs_seg[1:kmax,1:q_length], 1, q_length, cOff)
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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

function distPointNearKNN_fileout(
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
               
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
            
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
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1

                   q_length = min(qSize, length(qStart:qStop))
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)

                   kdtree = NearestNeighbors.KDTree(C[:,1:c_length], leafsize = 16)

                   idxs_n, dists_n = NearestNeighbors.knn(kdtree, Q[:,1:q_length], kmax, true)                   
            
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:kmax
                             idxs_seg[jj,ii] = idxs_n[ii][jj] + cOff
                             dist_seg[jj,ii] = dists_n[ii][jj]
                         end
                   
                   end
                   
                   kdtree = nothing
                   idxs_n = nothing
                   dists_n = nothing
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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

function distPointNearKNN(
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

             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = similar(idxs)
             dist_seg = similar(dist)
             
             
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
            
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
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)
            
                   cOff = cStart - 1

                   q_length = min(qSize, length(qStart:qStop))
                   c_length = min(cSize, length(cStart:cStop))
                   kmax = min(k, c_length)                   
                   
                   kdtree = NearestNeighbors.KDTree(C[:,1:c_length], leafsize = 16)

                   idxs_n, dists_n = NearestNeighbors.knn(kdtree, Q[:,1:q_length], kmax, true)                   
            
                   @inbounds for ii in 1:length(qStart:qStop)
                         for jj in 1:kmax
                             idxs_seg[jj,ii] = idxs_n[ii][jj] + cOff
                             dist_seg[jj,ii] = dists_n[ii][jj]
                         end
                   
                   end
                   
                   kdtree = nothing
                   idxs_n = nothing
                   dists_n = nothing
                   
                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
               
               if (qLocStop - qLocStart) == (qSize - 1)
                   copy2dArrPart!(idxs_out,qLocStart,qLocStop,idxs,1,qSize)
                   copy2dArrPart!(dist_out,qLocStart,qLocStop,dist,1,qSize)
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
     
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             distance = zeros(Float32,cSize,qSize)
             
             tempIndex = zeros(Int32,cSize,qSize)
             
             
             np = Threads.nthreads()
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
                   
                   kmax = min(k, c_length)

                   @views @inbounds Threads.@threads for l in 1:q_length
                        euclDistP(Q[:,1:q_length],C[:,1:c_length],distance,l,c_length,d)                   
                        idxs_seg[1:kmax,l] .= partialsortperm!(tempIndex[1:c_length,l],distance[1:c_length,l],1:kmax,initialized=false) .+ cOff
                        dist_seg[1:kmax,l] .= distance[tempIndex[1:kmax,l],l]
                        
                        nothing
                   end

                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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
           
         idxs = zeros(Int32, k, qSize)
         dist = zeros(Float32, k, qSize)
         
         idxs_seg = zeros(Int32, k, qSize)
         dist_seg = zeros(Float32, k, qSize)
  
         Indx = zeros(Int32, k, qSize)
         Dists = zeros(Float32, k, qSize)
         
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
        
               
               cOff = cStart - 1 
               
               q_length = min(qSize, length(qStart:qStop)) 
               c_length = min(cSize, length(cStart:cStop))

               kmax = min(k, c_length)               

               @views @inbounds Threads.@threads for l in 1:q_length
                    euclDistP(Q[:,1:q_length],C[:,1:c_length],distance,l,c_length,d)                   
                    idxs_seg[1:kmax,l] .= partialsortperm!(tempIndex[1:c_length,l],distance[1:c_length,l],1:kmax,initialized=false) .+ cOff
                    dist_seg[1:kmax,l] .= distance[tempIndex[1:kmax,l],l]
                    
                    nothing
               end
               
               if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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



function distPointKNN_rp_CPU(
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
             r::Int,
             P::Int
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
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
     
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             distance = zeros(Float32,cSize,qSize)
             
             tempIndex = zeros(Int32,cSize,qSize)
             
             
             np = Threads.nthreads()
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
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)

                   cOff = cStart - 1 
                   
                   q_length = min(qSize, length(qStart:qStop)) 
                   c_length = min(cSize, length(cStart:cStop))
                   
                   kmax = min(k, c_length)
                   
                   rpKNN_CPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,r,P,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
                   idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff

                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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

function distPointKNN_rp_CPU_fileout(
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
             r::Int,
             P::Int   
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
           
         idxs = zeros(Int32, k, qSize)
         dist = zeros(Float32, k, qSize)
         
         idxs_seg = zeros(Int32, k, qSize)
         dist_seg = zeros(Float32, k, qSize)
  
         Indx = zeros(Int32, k, qSize)
         Dists = zeros(Float32, k, qSize)
         
         distance = zeros(Float32,cSize,qSize)
         
         tempIndex = zeros(Int32,cSize,qSize) 
         
         np = Threads.nthreads()
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
            
            @inbounds @views for j in 1:cBlock
               load_file_wrapper!(filenameC,d,C, cStart:cStop)

               cOff = cStart - 1 
               
               q_length = min(qSize, length(qStart:qStop)) 
               c_length = min(cSize, length(cStart:cStop))

               kmax = min(k, c_length)               

               rpKNN_CPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,r,P,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
               idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff  

               if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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


function distPointKNN_cen_CPU(
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
             
             idxs = zeros(Int32, k, qSize)
             dist = zeros(Float32, k, qSize)
      
             idxs_seg = zeros(Int32, k, qSize)
             dist_seg = zeros(Float32, k, qSize)
             
             Indx = zeros(Int32, k, qSize)
             Dists = zeros(Float32, k, qSize)
             
     
             idxs_out = zeros(Int32, k, b)
             dist_out = zeros(Float32, k, b)
             
             distance = zeros(Float32,cSize,qSize)
             
             tempIndex = zeros(Int32,cSize,qSize)
             
             
             np = Threads.nthreads()
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
                
                q_length = min(qSize, length(qStart:qStop))
             
                
                @inbounds @views for j in 1:cBlock
                   load_file_wrapper!(filenameC,d,C, cStart:cStop)

                   cOff = cStart - 1 
                   
                    
                   c_length = min(cSize, length(cStart:cStop))
                   
                   kmax = min(k, c_length)
                   
                   cenKNN_CPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
                   idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff

                   if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
               else
                    lcount = 1
                     @inbounds for z in qLocStart:qLocStop
                         for w in 1:k
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

function distPointKNN_cen_CPU_fileout(
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
           
         idxs = zeros(Int32, k, qSize)
         dist = zeros(Float32, k, qSize)
         
         idxs_seg = zeros(Int32, k, qSize)
         dist_seg = zeros(Float32, k, qSize)
  
         Indx = zeros(Int32, k, qSize)
         Dists = zeros(Float32, k, qSize)
         
         distance = zeros(Float32,cSize,qSize)
         
         tempIndex = zeros(Int32,cSize,qSize) 
         
         np = Threads.nthreads()
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
            
            @inbounds @views for j in 1:cBlock
               load_file_wrapper!(filenameC,d,C, cStart:cStop)
        
               
               cOff = cStart - 1 
               
               q_length = min(qSize, length(qStart:qStop)) 
               c_length = min(cSize, length(cStart:cStop))

               kmax = min(k, c_length)               

               cenKNN_CPU!(Q[:,1:q_length],C[:,1:c_length],d,kmax,idxs_seg[1:kmax,1:q_length],dist_seg[1:kmax,1:q_length])
                   
               idxs_seg[1:kmax,1:q_length] .= idxs_seg[1:kmax,1:q_length] .+ cOff

               if kmax < k
                       dist_seg[kmax+1:k,1:q_length] .= typemax(Float32)
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
               
               GC.gc()
        
               
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
                            # Algorithm 2 == brute force GPU
                            # Algorithm 3 == brute force CPU
                            # Algorithm 4 == ParallelNeighbors
                            # Algorithm 5 == random projection CPU
                            # Algorithm 6 == random projection GPU
                            # Algorithm 7 == TI filtering CPU
                            # Algorithm 8 == TI filtering GPU
                            # Algorithm 9 == NearestNeighbors
                            # Algorithm 10 == Faiss
                            
      in_memory::Bool = false,
      r::Int = 3,
      P::Int = 5
      )
      
      nearestKN = zeros(Int32,k,nQ)
      distsKN =  zeros(Float32,k,nQ)
  
      nw = nworkers()
      np = Threads.nthreads()        
      
      b = ceil(Int64, nQ/nw)

      idxs_seg = zeros(Int32, k,b)
      dist_seg = zeros(Float32, k, b)
      
      placeholders = []
      
      if qSize > b 
         qSize = b
      end
      
      numJobs = max( 1, ceil(Int32, b/qSize) ) * nw
      progress = RemoteChannel(()->Channel{Tuple}(numJobs));
      
      jobBounds = []
      
      GPUs = Int(length(devices()))
      gpuID = 0
      
      println("Starting distributed kNN calculation of $nQ Query points from file \"$filenameQ\" and $nC Corpus points from file \"$filenameC\"")
      println(" Number of workers: $nw")
      println(" Number of threads: $np")
      if algorithm == 2 || algorithm == 4 || algorithm == 6 || algorithm == 8 || algorithm == 10
         
         println(" Number of GPUs: $GPUs")
      end
      println(" Total number of job segments: $numJobs")
      
      jobCount = min(numJobs, nw)
      
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
                elseif algorithm == 4
                     pl = @spawnat j distPointParKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true, gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                elseif algorithm == 5
                     pl = @spawnat j distPointKNN_rp_CPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P)
                     jobCount -= 1
                elseif algorithm == 6
                     pl = @spawnat j distPointKNN_rp_GPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs 
                elseif algorithm == 7
                     pl = @spawnat j distPointKNN_cen_CPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 8
                     pl = @spawnat j distPointKNN_cen_GPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs
                elseif algorithm == 9
                     pl = @spawnat j distPointNearKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 10
                     pl = @spawnat j distPointFaiss_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs                                 
                
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
                elseif algorithm == 4
                     pl = @spawnat j distPointParKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true,gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                elseif algorithm == 5
                     pl = @spawnat j distPointKNN_rp_CPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P)
                     jobCount -= 1
                elseif algorithm == 6
                     pl = @spawnat j distPointKNN_rp_GPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs          
                elseif algorithm == 7
                     pl = @spawnat j distPointKNN_cen_CPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 8
                     pl = @spawnat j distPointKNN_cen_GPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs
                elseif algorithm == 9
                     pl = @spawnat j distPointNearKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 10
                     pl = @spawnat j distPointFaiss(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs                                 
                end
         end
         
         push!(placeholders,pl)
         push!(jobBounds, (startP,stopP))
         
         if jobCount == 0 
             break
         end
         
         startP = min(stopP + 1, nQ)
         stopP = min(stopP + b, nQ)
      end
       
       jobCount = numJobs
       counter = 0
       if algorithm == 2 || algorithm == 4 || algorithm == 6 || algorithm == 8 || algorithm == 10
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
               
               locStart, locStop = popfirst!(jobBounds)
               
               jobCount -= 1
               
               load_bin_segment_file!(filename_indxs, k, idxs_seg, 1:b, Int32)
               load_bin_segment_file!(filename_dists, k, dist_seg, 1:b, Float32)
               
               if isfile(filename_indxs)
                  rm(filename_indxs)
               end
               if isfile(filename_dists)
                  rm(filename_dists)
               end
               
               nearestKN[1:k,locStart:locStop] .= idxs_seg[1:k,:]
               distsKN[1:k,locStart:locStop] .= dist_seg[1:k,:]
               
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
               
               locStart, locStop = popfirst!(jobBounds)
               
               jobCount -= 1

               nearestKN[1:k,locStart:locStop] .= idxs_seg[1:k,:]
               distsKN[1:k,locStart:locStop] .= dist_seg[1:k,:]
               
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
                            # Algorithm 2 == brute force GPU
                            # Algorithm 3 == brute force CPU
                            # Algorithm 4 == ParallelNeighbors
                            # Algorithm 5 == random projection CPU
                            # Algorithm 6 == random projection GPU
                            # Algorithm 7 == TI filtering CPU
                            # Algorithm 8 == TI filtering GPU
                            # Algorithm 9 == NearestNeighbors
                            # Algorithm 10 == Faiss
                            
      in_memory::Bool = false,
      r::Int = 3,
      P::Int = 5
      )

      nw = nworkers()
      np = Threads.nthreads()      
      
      b = ceil(Int64, nQ/nw)
      
      if in_memory == true
          idxs_seg =  zeros(Int32,k,b)
          dist_seg = zeros(Float32,k,b)
      end
     
      placeholders = []
      
      if qSize > b 
         qSize = b
      end
      
      
      numJobs = max( 1, ceil(Int32, b/qSize) ) * nw
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
      if algorithm == 2 || algorithm == 4 || algorithm == 6 || algorithm == 8 || algorithm == 10
         
         println(" Number of GPUs: $GPUs")
      end
      println(" Total number of job segments: $numJobs")
      
      jobCount = min(numJobs, nw)
      
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
                elseif algorithm == 4
                     pl = @spawnat j distPointParKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true, gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                elseif algorithm == 5
                     pl = @spawnat j distPointKNN_rp_CPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P)
                     jobCount -= 1
                elseif algorithm == 6
                     pl = @spawnat j distPointKNN_rp_GPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs
                elseif algorithm == 7
                     pl = @spawnat j distPointKNN_cen_CPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 8
                     pl = @spawnat j distPointKNN_cen_GPU_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs
                elseif algorithm == 9
                     pl = @spawnat j distPointNearKNN_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 10
                     pl = @spawnat j distPointFaiss_fileout(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs                      
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
                elseif algorithm == 4
                     pl = @spawnat j distPointParKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, true, gpuID)
                     gpuID = (gpuID + 1) % GPUs                     
                     jobCount -= 1
                elseif algorithm == 5
                     pl = @spawnat j distPointKNN_rp_CPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P)
                     jobCount -= 1
                elseif algorithm == 6
                     pl = @spawnat j distPointKNN_rp_GPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, r, P, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs                     
                elseif algorithm == 7
                     pl = @spawnat j distPointKNN_cen_CPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 8
                     pl = @spawnat j distPointKNN_cen_GPU(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs 
                elseif algorithm == 9
                     pl = @spawnat j distPointNearKNN(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k)
                     jobCount -= 1
                elseif algorithm == 10
                     pl = @spawnat j distPointFaiss(progress, filenameC, filenameQ, startP, stopP, cSize, qSize, nC, nQ, d, k, gpuID)
                     jobCount -= 1
                     gpuID = (gpuID + 1) % GPUs            
                end
         end
       
         push!(placeholders,pl)
         
        if jobCount == 0 
            break
        end
         
         startP = min(stopP + 1, nQ)
         stopP = min(stopP + b, nQ)
      end
       
       jobCount = numJobs
       counter = 0
       if algorithm == 2 || algorithm == 4 || algorithm == 6 || algorithm == 8 || algorithm ==10
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
               
               jobCount -= 1
           
               if counter == 0
                   store_bin_file(file_indxs_out, k, nQ, filename_indxs, false, Int32)
                   store_bin_file(file_dists_out, k, nQ, filename_dists, false, Float32)
               else
                   store_bin_file(file_indxs_out, k, nQ, filename_indxs, true, Int32)
                   store_bin_file(file_dists_out, k, nQ, filename_dists, true, Float32)
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
