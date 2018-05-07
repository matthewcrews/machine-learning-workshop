#I "packages/XPlot.GoogleCharts/lib/net45"
#I "packages/Newtonsoft.Json/lib/net40"
#I "packages/FSharp.Data/lib/net45"
#I "packages/Google.DataTable.Net.Wrapper/lib"
#I "packages/Suave/lib/net40"
#I "packages/FsAlg/lib"
#I "packages/DiffSharp/lib"
#I "packages/FSharp.Quotations.Evaluator/lib/net40"
#I "packages/MathNet.Numerics/lib/net40"
#I "packages/MathNet.Numerics.FSharp/lib/net45"

#r "FSharp.Data.dll"
#r "XPlot.GoogleCharts.dll"
#r "Google.DataTable.Net.Wrapper.dll"
#r "Newtonsoft.Json.dll"
#r "FSharp.Quotations.Evaluator.dll"
#r "Suave.dll"
#r "DiffSharp.dll"    
#r "FsAlg.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"
open System.IO
open Suave
open Suave.Web
open Suave.Http
open Suave.Operators
open Suave.Filters

module FsiAutoShow =
  let mutable running = false
  let charts = ResizeArray<string>()

  let displayHtml html = 
    if not running then
      let part = pathScan "/chart/%d" (fun n -> Successful.OK (charts.[n]))
      async { startWebServer defaultConfig part } |> Async.Start
      running <- true

    charts.Add(html)
    let url = sprintf "http://localhost:8083/chart/%d" (charts.Count-1)
    System.Diagnostics.Process.Start(url) |> ignore

  fsi.AddPrinter(fun (chart:XPlot.GoogleCharts.GoogleChart) ->
    chart.GetHtml () |> displayHtml
    "(Google Chart)")

let __<'T> : 'T = failwith "Not implemented!"