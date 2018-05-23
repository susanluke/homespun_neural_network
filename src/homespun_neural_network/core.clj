(ns homespun-neural-network.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M])
  (:gen-class))

(defn sum-rows
  [mat]
  (map (partial apply +)
       (m/rows mat)))

(defn vector-mean
  [vec]
  (let [m (count vec)]
    (->> vec
         (apply +)
         (* (/ 1 m)))))

(defn matrix-row-mean
  [mat]
  (let [m (-> mat m/shape second)]
    (->> mat
         m/rows
         (map (partial apply +))
         (M/* (/ 1 m)))))

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
        dW (matrix-row-mean (m/emul dZ X))
        db (vector-mean dZ)]
    {:dW dW
     :db db}))


(defn linear-regression
  [X Y {:keys [W b] :as parameters}
   learning-rate num-iterations]
  (if (= num-iterations 0)
    parameters
    (let [A     (forward-prop X W b)
          cost  (cost Y A)
          grads (back-prop A Y X)
          new-params (update-params parameters grads learning-rate)]
      (recur X Y new-params
             learning-rate (dec num-iterations))))
  )


(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
