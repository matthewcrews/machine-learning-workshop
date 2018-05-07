module NearestNeighbor =

    let euclDistance (xs:float[]) (ys:float[]) =
        Array.map2 (fun x y -> pown (x-y) 2) xs ys
        |> Array.sum
        |> sqrt

    let absDistance (xs:float[]) (ys:float[]) =
        (xs, ys)
        ||> Array.map2 (fun x y -> abs (x-y))
        |> Array.sum 

    let classify examples distance item =
        examples
        |> Array.minBy (fun (_,observation) ->
            distance item observation)
        |> fst
        