module NearestNeighbor =

    let distance (xs:float[]) (ys:float[]) =
        Array.map2 (fun x y -> pown (x-y) 2) xs ys
        |> Array.sum
        |> sqrt

    let classify examples item =
        examples
        |> Array.minBy (fun (label,observation) ->
            distance item observation)
        |> fst
        