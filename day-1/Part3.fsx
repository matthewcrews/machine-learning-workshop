// The 'FsLab.fsx' script loads the XPlot charting library and other 
// dependencies. It is a smaller & simpler version of the FsLab 
// package that you can get from www.fslab.org.

#load "FsLab.fsx"
open FsLab
open System
open System.IO
open XPlot.GoogleCharts
open DiffSharp.Numerical

// ----------------------------------------------------------------------------
// PART 2. Here, we build a simple neural network (with just a single neuron)
// to distinguish between languages. This has better results than the
// solution in Part 1, but it is a bit more work!
// We will focus on distinguishing just between 2 languages.
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// STEP #1: First, we need a few things form the first step, most
// importantly the 'getFeatureVector' function that you wrote

let cleanDir = __SOURCE_DIRECTORY__ + "/clean/"
let featuresFile = __SOURCE_DIRECTORY__ + "/features.txt"
let features = File.ReadAllLines(featuresFile)

let getFeatureVector text = 

    let counts = 
        text
        |> Seq.pairwise
        |> Seq.map (fun (c1,c2) -> string c1 + string c2)
        |> Seq.countBy id
        |> Seq.toArray

    let total = (String.length text - 1) |> float
    let countsLookup = counts |> dict

    features   
    |> Array.map (fun pair ->
        if countsLookup.ContainsKey pair
        then float countsLookup.[pair] / total
        else 1e-10)


// ----------------------------------------------------------------------------
// The KNN model we used previously didn't really 'learn'; it scans the whole
// dataset every time it makes a prediction.
// What we will do here instead is learn a model once, using all the data,
// and create a compact model, a 'neuron', which we can then reuse. 
// The neuron will predict a value between 0 and 1. A value of 0.5 indicates
// 'I don't know'; the closer to 1.0 the value, the more certain the model
// is that the text is English, the closer to 0.0, the more certain the model
// is that the text is French.
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// STEP #2: We will start by creating a 'neuron' by hand, without any
// machine learning, to understand how the model works. We will focus on the
// learning part next.

// The sigmoid function is used to transform the output of a neuron
// into the [0,1] interval.
// See wikipedia for more information on the sigmoid function:
// https://en.wikipedia.org/wiki/Sigmoid_function

let sigmoid arg = 1. / (1. + exp (-arg))
Chart.Line [ for x in -10.0 .. 0.01 .. 10.0 -> x, sigmoid x ]

// For training the neural network, we need to add 1.0 to the 
// feature vector to account for the bias
let prependOne arr = Array.append [| 1.0 |] arr

// The prediction function takes all inputs, multiplies it by weights
// and then applies the sigmoid function to the sum of the result:
//
//             f1      f2      f3
//               \     | w2  /
//              w1 \   |   / w3
//                   \ | /
//   sigmoid (f1*w1 + f2*w2 + f3*w3)

let predict weights features = 
    (weights,features)
    ||> Array.map2 (fun w f -> w * f)
    |> Array.sum
    |> sigmoid

// let's try this on 2 examples, using 'flat' weights

let fakeWeights = Array.init (features.Length + 1) (fun i -> 0.0)

let englishSample = 
    "do you enjoy learning by getting your hands dirty and getting stuck into new concepts and ideas"
    |> getFeatureVector
    |> prependOne

let frenchSample = 
    "de mysterieux bateaux fantomes avec des cadavres a bord decouverts sur les cotes du japon"
    |> getFeatureVector
    |> prependOne
    
// what does our neuron predict for our 2 examples?
predict fakeWeights englishSample

predict fakeWeights frenchSample



// ----------------------------------------------------------------------------
// STEP #3: our neuron is currently pretty dumb - regardless of the input,
// it predicts 0.5, that is, 'I don't know'.
// We will now manually create weights that should do better:
// we will put a positive weight for 'ng', which is frequent in English, 
// and negative weights on 'de', which is frequent in French.


// TODO: create weights manualWeights, with a positive value 
// (5.0 for instance) in the position corresponding to 'ng', 
// and negative (-5.0 for instance) for 'de'.

// create empty weights, with 0.0 everywhere
let emptyWeights () = 
    Array.init (features.Length + 1) (fun _ -> 0.0)

// replace weights for a pair of characters
let setWeightFor (pair,weight) (weights:float[]) =
    let index = Array.IndexOf(features, pair)
    weights 
    |> Array.mapi (fun i w ->
        if i = index + 1 then weight else w)

let manualWeights = 
    emptyWeights ()
    |> setWeightFor (__, __)
    |> setWeightFor (__, __)

 
// what does our neuron predict now?

predict manualWeights __ 

predict __ __


// TODO: play with weights to see if you can improve predictions,
// putting positive values on pairs indicative of English, and  
// negative ones for pairs common in French.



// ----------------------------------------------------------------------------
// STEP #4: introduction to Gradient Descent.
// Finding weights that work by adjusting them manually is hopeless. What we 
// want now is to let the machine learn weights that work well, using the
// whole sample.
// To do that, we will use an algorithm called Gradient Descent, which will 
// help us find the weights that give us the smallest prediction error across
// our entire training set.
// Before applying it to our problem, we will spend a bit of time explaining
// Gradient Descent, which is a powerful technique applicable to many problems.


// ----------------------------------------------------------------------------
// DEMO: To implement the training for our neuron, we'll use a library
// DiffSharp that lets us automatically differentiate functions. Let's
// look at a few examples of how this works! 

// Define a sample function & draw a chart:

let sinSqrt x = sin (3.0 * sqrt x)
Chart.Line [ for x in 0.0 .. 0.01 .. 10.0 -> x, sinSqrt x ]

// over 0 .. 10 this function has 2 maximums, and 1 minimum


// We can use 'diff', a function from DiffSharp, to 
// differentiate any function! Let's do this on 'sinSqrt', 
// and plot the charts

let sinSqrtDiff = diff sinSqrt

Chart.Line
  [ [ for x in 0.0 .. 0.01 .. 10.0 -> x, sinSqrt x ]
    [ for x in 0.0 .. 0.01 .. 10.0 -> x, sinSqrtDiff x ] ]
|> Chart.WithLabels ["function"; "derivative"]

// We can see two things on that chart:
// 1) the derivative is equal to zero everywhere the
// function has a maximum or a minimum
// 2) when the function increases, the derivative is
// positive, and vice-versa.

// This gives us a strategy to find the minimum of a function:
// Start with an arbitrary value x0
// If the derivative f'(x0) is positive, the function 
// is increasing: decrease x by 'a little'
// If the derivative f'(x0) is negative, the function
// is decreasing: increase x by 'a little'
// If the derivative is close to 0, move slowly, 
// because we are close to a min or max, otherwise,
// move more aggressively: take steps proportional to derivative.


// The resulting algorithm is called the 'gradient descent':
// update x1 <- x0 - eta * f'(x0)
// eta is a parameter corresponding to 'a little'.
// Repeat until the value xn barely changes. 

// DEMO: gradient descent on SinSqrt, starting from x0 = 6.0

// update to the next value of x, based on gradient:
let gradient_update eta x =
    let derivative = sinSqrtDiff x
    x - eta * derivative

// compute each steps of a gradient descent, starting
// at value x0, with a given value of eta, for a fixed
// number of iterations:
let steps iters eta x0 =

    let rec iterate iter acc x =
        // next value of x
        let x' = gradient_update eta x
        // append (x,f(x)) to accumulator
        let acc = (x',sinSqrt x') :: acc
        if iter = 0
        then acc |> List.rev
        else iterate (iter-1) acc x'
    
    iterate iters [(x0,sinSqrt x0)] x0 

// TODO: run a Gradient Descent for 20 iterations,
// starting from x0 = 6.0, with eta = 0.5
let gradient_example = steps __ __ __


Chart.Scatter
  [ [ for x in 0.0 .. 0.01 .. 10.0 -> x, sinSqrt x ]
    gradient_example ]
|> Chart.WithLabels ["function"; "descent"]


// TODO: try running gradient descent with: 
// different values of eta: what is the result?
// different values of x0: what is the result?


// ----------------------------------------------------------------------------
// STEP #5: MATH ALERT - more Gradient Descent!
// The previous example illustrates how the idea works on a simple function, 
// with one argument only. However, it extends to functions taking as many
// arguments as you need - and DiffSharp supports that scenario as well, 
// using grad instead of diff.
// NOTE: if you are not clear on every detail in this section, don't worry!


// We can also differentiate functions of multiple parameters,
// but they need to take parameters as arrays of floats
// Example: plot the 'mexican hat' function
let mexicanHat (arr:float[]) = 
    let x, y = arr.[0], arr.[1]
    let r = sqrt(x*x + y*y) + 1e-10
    (sin r) / r;

Chart.Line [ for x in -10.0 .. 0.01 .. 10.0 -> x, mexicanHat [|x;x|] ]

// Mexican hat is a two-dimensional function. To get multi-variable
// differentiation, we use the 'grad' function. Then we can see 
// the direction (how much is the function going up/down) at
// various points in both of the two directions.

// Computing gradient of the 'mexican hat' function:
let mexicanHatDiff = grad mexicanHat

// Evaluating the gradient at specific locations
mexicanHatDiff [| 0.0; 0.0 |]
mexicanHatDiff [| 1.5; 1.5 |]
mexicanHatDiff [| 3.2; 3.2 |]
mexicanHatDiff [| 0.0; 6.0 |]


// ----------------------------------------------------------------------------
// STEP #6: Preparing the data
// Now that we have a model (our neuron) and a strategy to learn weights
// (gradient descent), let's apply this to our problem.

// This time, we load data for two languages and we split them into individual
// pages (training samples). We then build training data where we have the
// expected result (1.0 for one language, 0.0 for the other language) together
// with the feature vectors.

// Split data into individual pages for each of the languages
let allTrainingData = 
    Directory.GetFiles(cleanDir, "*.txt")
    |> Array.map (fun file ->
        let lang = Path.GetFileNameWithoutExtension(file)
        let text = File.ReadAllText(file)
        let sentenceFeatures = 
            text.Split [|'\n'|]
            |> Array.map getFeatureVector
        lang, sentenceFeatures)
    |> dict

// Get feature vector for two languages that we want to recognize
// For the first language (English), the true label is 1
// For the second language (French), the true label is 0
let lang1 = allTrainingData.["English"] |> Array.map (fun v -> 1.0, prependOne v)
let lang2 = allTrainingData.["French"] |> Array.map (fun v -> 0.0, prependOne v)

// Build the training data set by appending the two arrays
let trainingData = Array.append lang1 lang2


// ----------------------------------------------------------------------------
// STEP #7: Learning English & French!

// We are now ready to train our neuron. We will use Gradient Descent to
// adjust the weights and minimize the prediction error.
// The error function takes the training data & current weights and calculates
// how good our result is. That is, for all the items in the input training data, 
// we have the expected label and features.
// For each language features in 'trainingData', we call 'predict weights features' 
// to calculate the predicted label. Then we calculate the Euclidean 
// distance between the predicted label and the true label.
// We sum this over all the training data.
let error trainingData weights = __


// The initial weights for the neuron are generated randomly (this gives 
// bad results :-), but we need something to start the training!)

let seed = 12345678

let initialWeights = 
    let rnd = System.Random(seed)
    Array.init (features.Length + 1) (fun _ -> rnd.NextDouble())

// What is the current error using the random weights?
error trainingData initialWeights


// ----------------------------------------------------------------------------
// STEP #8: Training the neural network.
// To train the neural network, we need to improve the weights. To do this
// we'll calculate the "gradient" of the 'error' function with respect
// to the weights. Then we adapt the weights by "jumping"
// in the right direction by a constant called "eta".

let eta = 0.2

// Compute derivative of the 'error' function, using 'grad'. 
// The 'errorGradient' should be a function taking weights as its
// input (you can use partial function application!)
let errorGradient = grad (error trainingData)

// Compute the initial error gradient
let gradient = __


// Now we want to minimize the error function which means we'll be going
// against the gradient. 
// We have the gradient for the initial value of weights,
// and we want to calculate new weights such that:
//   newWeight[i] = oldWeight[i] - gradient[i]*eta
// We can nicely do this using the 'Array.map2' function!
let newWeights =
    (initialWeights,gradient) 
    ||> Array.map2 (fun w g -> w - eta * g)

// Assuming your training improved the weights, 
// the error should be getting smaleler!
error trainingData initialWeights
error trainingData newWeights


// Now we just need to run the weight adaptation that you wrote in loop.
// The following is a recursive function that counts how many times it runs
// and it adapts the weights 5000 times (you can try changning this too).
// You just need to fill in the part that calculates 'newWeights'
let rec gradientDescent steps weights =
  if steps > 5000 then 
    weights
  else
    let gradient = errorGradient weights
    let newWeights = __
    gradientDescent (steps+1) newWeights


// After the training, the error should be much smaller
let trained = gradientDescent 0 initialWeights
error trainingData trained


// Try it out on our 2 examples 
predict trained englishSample

predict trained frenchSample

// What are the pairs of letters most indicative of
// English? Of French? What pairs do not seem to matter?


// ----------------------------------------------------------------------------
// OPEN QUESTIONS


// 1. Change the 'gradientDescent' function to also return the error value
//    in each iteration.


// 2. Plot the error after training to see how long it takes until the value
//    converges. 
//    - How does this depend on the step size 'eta'?


// 3. Observe how the error during training changes when we use the neuron
//    to distinguish the two most different languages (identified in Part 1).
//    How does the error look when we train the neuron to distinguish the two
//    most similar languages? 
//    What would you do to get better performance?
