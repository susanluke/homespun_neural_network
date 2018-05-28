(ns homespun-neural-network.neural-network
  (:require [homespun-neural-network.matrix-utils :as hnn-m]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]
            [clojure.core.matrix.random :as mr]))

;; TODO - not sure it's nice to store grads & net params in
;; maps.  Maybe change to storing in vectors and using numerical
;; indexes.

;; Hyperparameters
(def initial-weight-scale 0.0001)

(defn num-layers [m]
  (count (:layer-sizes m)))

(defn relu [m] (m/emap #(max 0 %) m))
(defn relu-grad [m] (m/emap #(if (< 0 %) 1 0) m))
(defn logistic-grad [m] (m/emap #(* % (- 1 %)) m))

(def activation-fns {:relu {:fn relu
                            :dfn relu-grad}
                     :sigmoid {:fn m/logistic
                               :dfn logistic-grad}
                     :identity {:fn identity ; only for test purposes
                                :dfn (constantly 1)}})

(defn make-key [k n] (keyword (str (name k) n)))

(defn init-net-params
  "Takes:
  layer-sizes - list containing number of nodes in each layer
  fns - list containing activation fn for each layer
  Returns:
  net-params - a map containing
      :layer-sizes - vector containing number of nodes in each layer
      :W - a vector containing nil, W1, W2, ... for each layer
      :b - a vector containing nil, b1, b2, ... for each layer
  NB initial value in :W, :b is nil to make indexing by layer simpler.
  "
  [layer-sizes fns]
  (let [num-layers (count layer-sizes)]
    (loop [net-params {:layer-sizes layer-sizes
                       :fns fns
                       :W [nil]
                       :b [nil]}
           idx 1]
      (if (= num-layers idx)
        net-params
        (let [random-numbers (mr/randoms idx)
              W-rows (nth layer-sizes idx)
              W-cols (nth layer-sizes (dec idx))
              W (-> (take (* W-rows W-cols) random-numbers)
                    (m/reshape [W-rows W-cols])
                    (M/* initial-weight-scale))
              b 0] ; set b to 0, rely on broadcasting to scale up
          (recur (assoc net-params
                        :W (conj (:W net-params) W)
                        :b (conj (:b net-params) b))
                 (inc idx)))))))

;; TODO: Maybe we can just map over the arrays, rather than loop/recur?
(defn update-net-params
  [{:keys [layer-sizes fns] :as net-params} grads learning-rate]
  (let [num-layers (num-layers net-params)]
    (loop [new-net-params {:layer-sizes layer-sizes
                           :fns fns
                           :W [nil]
                           :b [nil]}
           idx 1]
      (if (= num-layers idx)
        new-net-params
        (let [W (nth (:W net-params) idx)
              b (nth (:b net-params) idx)
              dW (nth (:dW grads) idx)
              db (nth (:db grads) idx)
              W-new (M/- W (M/* learning-rate dW))
              b-new (M/- b (M/* learning-rate db))]
          (recur (assoc new-net-params
                        :W (conj (:W new-net-params) W-new)
                        :b (conj (:b new-net-params) b-new))
                 (inc idx)))))))

;; TODO: poss re-write as reduce?
(defn forward-prop
  [X net-params]
  (let [num-layers (num-layers net-params)]
    (loop [state {:A [X]
                  :Z [nil]}
           idx 1]
      (if (= num-layers idx)
        state
        (let [W      (nth (:W net-params) idx)
              b      (nth (:b net-params) idx)
              A-prev (nth (:A state) (dec idx))
              fn-key (nth (:fns net-params) idx)
              a-fn   (get-in activation-fns [fn-key :fn])
              Z      (->> A-prev
                          (m/dot W)
                          (M/+ b))
              A      (a-fn Z)]
          (recur (assoc state
                        :Z (conj (:Z state) Z)
                        :A (conj (:A state) A))
                 (inc idx)))))))

(defn cost
  "Y & A are matrices of shape [1 m]
  Returns the logisitic cost based on formula:
  L = (-1*m).sum[y.log(a) + (1-y).log(1-a)]"
  [Y A]
  (let [m (-> Y m/shape second)] ; number of columns in Y
    (* (/ -1 m)
       (m/esum (M/+
                (M/* Y (m/log A))
                (M/* (M/- 1 Y) (m/log (M/- 1 A))))))))


;; TODO: need to refactor based on new format for
;; net-params and state
(defn back-prop
  "Given current activation state for the network, calculates and
  returns gradients for each weight and bias"
  [X Y state net-params]
  (let [layers (num-layers net-params)
        m (-> Y m/shape second)
        A ((make-key :A (dec layers)) state)
        dZl (m/esum (M/- A Y))]
    (loop [l (dec layers)
           grads {(make-key :dZ l) dZl}]
      (if (= 0 l)
        grads
        (recur
         (dec l)
         (let [dZ     ((make-key :dZ l) grads)
               A-prev ((make-key :dA (dec l)) state)
               W      ((make-key :W l) net-params)
               Z-prev ((make-key :Z (dec l)) state)
               dfn-prev (:dfn ((make-key :fn (dec l)) net-params))
               dW (-> dZ
                      (m/dot (m/transpose A-prev))
                      hnn-m/matrix-row-mean)
               db (hnn-m/matrix-row-mean dW)
               dA-prev (->> dZ
                            (m/dot (m/transpose A))
                            hnn-m/matrix-row-mean)
               dZ-prev (->> dA-prev
                            (M/* (m/emap dfn-prev Z-prev)))]
           (assoc grads
                  (make-key :dW l) dW
                  (make-key :db l) db
                  (make-key :dA (dec l)) dA-prev
                  (make-key :dZ (dec l)) dZ-prev)))))))

(defn neural-network
  [X Y net-params learning-rate num-iterations]
  (if (= 0 num-iterations)
    net-params
    (let [state (forward-prop X net-params)
          A ((make-key :A (num-layers net-params)) state)
          cost (cost Y A)
          grads (back-prop X Y state net-params)
          new-net-params (update-net-params net-params grads learning-rate)]
      (recur X Y net-params learning-rate num-iterations))))
