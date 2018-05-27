(ns homespun-neural-network.logistic-regression
  (:require [homespun-neural-network.matrix-utils :as hnn-m]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]))

(defn init-params
  "Returns initial values for W and b, given the number of dimensions
  in x"
  [n-x]
  {:W (m/zero-matrix n-x 1)
   :b 0})

(defn update-params
  "Given current weight and bias parameters, their gradients
  and learning rate, return new weights and bias"
  [{:keys [W b] :as params}
   {:keys [dW db] :as grads}
   learning-rate]
  {:W (M/- W (M/* learning-rate dW))
   :b (- b (* learning-rate db))})

(defn forward-prop
  "Returns A for given X, W & b"
  [X W b]
  (->> X
       (m/dot (m/transpose W))
       (m/add b)
       m/logistic))

(defn cost
  "Returns the logistic cost for
   Y (expected output) and A (actual output)"
  [Y A]
  (let [m (count Y)]
    (* (/ -1 m) (m/esum (M/+ (M/* Y ; term relevant when Y=1
                                  (m/log A))
                             (M/* (m/sub 1 Y) ; term relevant when Y=0
                                  (m/log (m/sub 1 A))))))))

(defn back-prop
  "Given A, Y & X, returns gradients for W and b"
  [A Y X]
  (let [m (count Y)
        dZ (m/sub A Y)
        dW (hnn-m/matrix-row-mean (M/* dZ X))
        _ (println "[back-prop] dZ:" dZ)
        db (hnn-m/vector-mean dZ)]
    {:dW dW
     :db db}))

(defn linear-regression
  "Performs of linear regression for:
  X - input
  Y - desired output
  parameters - network weights and bias
  learning-rate - value to scale gradient by when adjusting weights & bias
  num-iterations - number of iterations

  Returns learned network parameters as a map in the form:
  {:W weights :b bias}"
  [X Y {:keys [W b] :as parameters}
   learning-rate num-iterations]
  (if (= num-iterations 0)
    parameters
    (let [A     (forward-prop X W b)
          cost  (cost Y A)
          grads (back-prop A Y X)
          new-params (update-params parameters grads learning-rate)]
      (recur X Y new-params
             learning-rate (dec num-iterations)))))
