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
(defn logistic-grad
  [m]
  (m/emap #(* (m/logistic %)
              (- 1 (m/logistic %)))
          m))

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
              b (m/zero-matrix W-rows 1)]
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
  (let [num-layers (num-layers net-params)
        m (-> X m/shape second)]
    (loop [state {:A [X]
                  :Z [nil]}
           idx 1]
      (if (= num-layers idx)
        state
        (let [_ (println "[FP]LAYER:" idx)
              W      (nth (:W net-params) idx)
              b      (nth (:b net-params) idx)
              A-prev (nth (:A state) (dec idx))
              fn-key (nth (:fns net-params) idx)
              a-fn   (get-in activation-fns [fn-key :fn])
              _ (println "[FP]W:" (m/shape W) W)
              _ (println "[FP]b:" (m/shape b) b)
              _ (println "[FP]A-prev:" (m/shape A-prev) A-prev)
              Z      (->> A-prev
                          (m/dot W)
                          (M/+ (apply (partial m/join-along 1) ; broadcast b out to match Z dims
                                      (repeat m b))))
              _ (println "[FP]Z:" (m/shape Z) Z)
              A      (a-fn Z)]
          (recur (assoc state
                        :Z (conj (:Z state) Z)
                        :A (conj (:A state) A))
                 (inc idx)))))))

(defn cost
  "Y & A are matrices of shape [1 m]
  Returns the logistic cost based on formula:
  L = (-1*m).sum[y.log(a) + (1-y).log(1-a)]"
  [Y A]
  (let [m (-> Y m/shape second)] ; number of columns in Y
    (* (/ -1 m)
       (m/esum (M/+
                (M/* Y (m/log A))
                (M/* (M/- 1 Y) (m/log (M/- 1 A))))))))

(defn cost-grad
  "Returns d(cost(Y,A)/dA
  returns a matrix of [[1]]"
  [Y A]
  (hnn-m/matrix-row-mean-keep-dims (M/- (M// Y A)
                                        (M// (M/- 1 Y)
                                             (M/- 1 A)))))

;; TODO - maybe just calc dA prior to loop/recur
;; then need to calc dZ dW dB and dA-prev
(defn back-prop
  "Given current activation state for the network, calculates and
  returns gradients for each weight and bias"
  [X Y net-params state]
  (let [layers (num-layers net-params)
        m (-> Y m/shape second)
        A-l (last (:A state))
        dZ-l [[(m/esum (M/- A-l Y))]]]
    (loop [l (dec layers)
           ;; use lists so conj adds to the head
           grads {:dZ (list dZ-l)
                  :dA '(nil)  ; don't bother calculating dJ/dA
                  :dW '()
                  :db '()}]
      (if (= 0 l)
        (assoc grads
               ;; Add in dW-0 and db-0 so indexes match up
               :dW (conj (:dW grads) nil)
               :db (conj (:db grads) nil))
        (recur
         (dec l)
         (let [;; retrieve relevant parameters
               _ (println "---")
               _ (println "[BP]LAYER " l)
               dZ     (first (:dZ grads))
               W      (nth (:W net-params) l)
               A-prev (nth (:A state) (dec l))
               Z-prev (nth (:Z state) (dec l))
               fn-key-prev (nth (:fns net-params) (dec l))
               dfn-prev (get-in activation-fns [fn-key-prev :dfn])
               ;; perform differential calculations
               _ (println "dZ:" (m/shape dZ) dZ)
               _ (println "W:" (m/shape W))
               _ (println "A-prev:" (m/shape A-prev))
               _ (println "Z-prev:" (m/shape Z-prev))

               _ (println "dot p1:" (hnn-m/matrix-row-mean-keep-dims A-prev))
               _ (println "dot p2:" (m/transpose dZ))

               dW (m/dot
                   dZ
                   (-> A-prev hnn-m/matrix-row-mean-keep-dims m/transpose))
               _ (println "dW:" (m/shape dW) dW)
               db (hnn-m/matrix-row-mean-keep-dims dZ)
               _ (println "db:" (m/shape db) db)
               dA-prev (if (= l 1)
                         nil ; hack to avoid going back too far
                         (m/dot (m/transpose W) dZ))
               _ (println "dA-prev:" (m/shape dA-prev) dA-prev)
               dZ-prev (if (= l 1)
                         nil ; hack to avoid going back too far
                         (M/* (hnn-m/matrix-row-mean-keep-dims (m/emap dfn-prev Z-prev))
                              dA-prev))
               _ (println "dZ-prev:" (m/shape dZ-prev) dZ-prev)]
           (assoc grads
                  :dW (conj (:dW grads) dW)
                  :db (conj (:db grads) db)
                  :dA (conj (:dA grads) dA-prev)
                  :dZ (conj (:dZ grads) dZ-prev))))))))

(defn neural-network
  [X Y net-params learning-rate num-iterations]
  (if (= 0 num-iterations)
    net-params
    (let [state (forward-prop X net-params)
          A ((make-key :A (num-layers net-params)) state)
          cost (cost Y A)
          grads (back-prop X Y state net-params)
          new-net-params (update-net-params net-params
                                            grads
                                            learning-rate)]
      (recur X Y net-params learning-rate num-iterations))))

(comment (defn back-prop
           "Given current activation state for the network, calculates and
  returns gradients for each weight and bias"
           [X Y net-params state]
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
                               hnn-m/matrix-row-mean-keep-dims)
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
                           (make-key :dZ (dec l)) dZ-prev))))))))
