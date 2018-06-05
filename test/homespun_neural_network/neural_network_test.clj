(ns homespun-neural-network.neural-network-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [homespun-neural-network.neural-network :as sut]))

(defn almost=
  [n1 n2]
  (let [diff (Math/abs (- n1 n2))]
    (< diff 1e-8)))

;; Sample data
(def net-params  {:layer-sizes [2 3 2]
                  :W [nil
                      [[1 2]
                       [4 5]
                       [6 7]]
                      [[100 200 350]
                       [300 400 450]]]
                  :b [nil
                      [[1]
                       [2]
                       [3]]
                      [[4]
                       [5]]]
                  :fns (repeat 3 :identity)})

(deftest relu-test
  (are [in out] (= (sut/relu in) out)
    -10 0
    -1  0
    0   0
    1   1
    10  10))

(deftest relu-grad
  (are [in out] (= (sut/relu-grad in) out)
    -10 0
    -1  0
    0   0
    1   1
    10  1))

(deftest logistic-grad-test
  (are [in out] (almost= (sut/logistic-grad in) out)
    0 0.25
    100 0.0
    -100 0.0))

(deftest init-net-params-test
  (let [{:keys [W b]} (sut/init-net-params [2 3 1]
                                           (repeat 3 :identity))]
    (is (= [3 2] (m/shape (nth W 1))))
    (is (= [1 3] (m/shape (nth W 2))))
    (is (= [[0.0] [0.0] [0.0]] (nth b 1)))
    (is (= [[0.0]] (nth b 2)))))

(deftest update-net-params-test
  ;; TODO: I don't think those layer sizes are consistent with
  ;; :W and :b values
  (let [net-params {:layer-sizes [2 3 2]
                    :W [nil
                        [[1 2 3]
                         [4 5 6]]
                        [[100 200]
                         [300 400]
                         [500 600]]]
                    :b [nil
                        [[1]
                         [2]
                         [3]]
                        [[4]
                         [5]]]}
        grads {:dW [nil
                    [[-10 -20 -30]
                     [40 50 60]]
                    [[-1 -2]
                     [3 4]
                     [5 6]]]
               :db [nil
                    [[-10]
                     [20]
                     [-30]]
                    [[40]
                     [-50]]]}
        learning-rate 0.5
        {:keys [W b]} (sut/update-net-params net-params
                                             grads
                                             learning-rate)]
    (is (= [[6.0 12.0 18.0]
            [-16.0 -20.0 -24.0]]
           (nth W 1)))
    (is (= [[100.5 201.0]
            [298.5 398.0]
            [497.5 597.0]]
           (nth W 2)))
    (is (= [[6.0]
            [-8.0]
            [18.0]]
           (nth b 1)))
    (is (= [[-16.0]
            [30.0]]
           (nth b 2)))))

(deftest forward-prop-test
  (testing "basic forward prop"
    (comment (let [X [[1] [2]]
                   state (sut/forward-prop X net-params)]
               (is (= (:A state)
                      [X
                       [[6] [16] [23]]
                       [[11854] [18555]]]))
               (is (= (:Z state)
                      [nil
                       [[6] [16] [23]]
                       [[11854] [18555]]]))))))

(defn matrix->vector
  [m]
  (let [size (apply * (m/shape m))]
    (m/reshape m [size])))

(defn net-params->vector
  "Converts W & b network parameters into a single vector, for use
  in grad check"
  [net-params]
  ;; NB, drop initial nil in :W and :b
  (->> (interleave (rest (:W net-params)) (rest (:b net-params)))
       (map matrix->vector)
       (apply m/join)))

;; TODO need to reverse grads - prob in back-prop
(defn grads->vector
  [grads]
  (->> (interleave (rest (:dW grads)) (rest (:db grads)))
       (map matrix->vector)
       (apply m/join)))

(defn vector->net-params
  [v layer-sizes fns]
  (let [num-layers (count layer-sizes)]
    (loop [net-params {:layer-sizes layer-sizes
                       :fns fns
                       :W [nil]
                       :b [nil]}
           layer 1
           v v]
      (if (= num-layers layer)
        net-params
        (let [num-nodes (nth layer-sizes layer)
              num-nodes-prev (nth layer-sizes (dec layer))
              W-shape [num-nodes num-nodes-prev]
              b-shape [num-nodes 1]
              W-num (apply * W-shape)]
          (recur
           (assoc net-params
                  :W (conj (:W net-params) (m/reshape v W-shape))
                  :b (conj (:b net-params) (m/reshape
                                            (drop W-num v)
                                            b-shape)))
           (inc layer)
           (drop (+ W-num num-nodes) v)))))))

(defn grad-approx
  [X Y {:keys [layer-sizes fns] :as net-params} epsilon]
  (let [net-params-vector (net-params->vector net-params)]
    (for [i (range  (count net-params-vector))]
      (let [theta-i     (nth net-params-vector i)
            theta-plus  (assoc net-params-vector i (+ theta-i epsilon))
            theta-minus (assoc net-params-vector i (- theta-i epsilon))
            state-plus  (sut/forward-prop X
                                          (vector->net-params
                                           theta-plus
                                           layer-sizes fns))
            state-minus (sut/forward-prop X
                                          (vector->net-params
                                           theta-minus
                                           layer-sizes fns))
            J-plus      (sut/cost Y (last (:A state-plus)))
            J-minus     (sut/cost Y (last (:A state-minus)))]
        (/ (- J-plus J-minus)
           (* 2 epsilon))))))


(deftest grad-check-single-training-case-test
  (testing
      "Checking the gradient manually is only slightly different
       from that calculated by back-prop for a single test case"
    (let [net-params {:layer-sizes [2 3 1],
                      :fns [:identity :relu :sigmoid],
                      :W
                      [nil
                       [[7.308781907032909E-5 4.100808114922017E-5]
                        [2.077148413097171E-5 3.327170559595112E-5]
                        [9.677559094241208E-5 6.1171822657613E-7]]
                       [[7.311469360199059E-5
                         9.014476240300544E-5
                         4.9682259343089074E-5]]],
                      :b [nil [[0.0] [0.0] [0.0]] [[0.0]]]}
          X [[1] [1000]]
          Y [[0]]
          state (sut/forward-prop X net-params)
          _ (println "state:" state)
          grads (sut/back-prop X Y net-params state)
          epsilon 1e-7
          grads-approx (grad-approx X Y net-params epsilon)]
      (println "grads-approx: " grads-approx)
      (println "grads:" grads)
      (println "----")
      (println "grads-approx2: " grads-approx)
      (println "grads->vector" (grads->vector grads))
      (println "grads2:" grads)
      (is (every? true?
                  (map almost= (grads->vector grads) grads-approx))))))

(deftest grad-check-two-training-cases-test
  (comment (testing
               "Checking the gradient manually is only slightly different
       from that calculated by back-prop for a two test cases"
             (let [net-params {:layer-sizes [2 3 1],
                               :fns [:identity :relu :sigmoid],
                               :W
                               [nil
                                [[7.308781907032909E-5 4.100808114922017E-5]
                                 [2.077148413097171E-5 3.327170559595112E-5]
                                 [9.677559094241208E-5 6.1171822657613E-7]]
                                [[7.311469360199059E-5
                                  9.014476240300544E-5
                                  4.9682259343089074E-5]]],
                               :b [nil [[0.0] [0.0] [0.0]] [[0.0]]]}
                   X [[1 3] [1000 200000000]]
                   Y [[0 1]]
                   state (sut/forward-prop X net-params)
                   _ (println "state:" state)
                   grads (sut/back-prop X Y net-params state)
                   epsilon 1e-7
                   grads-approx (grad-approx X Y net-params epsilon)]
               (println "grads-approx: " grads-approx)
               (println "grads:" grads)
               (println "----")
               (println "grads-approx2: " grads-approx)
               (println "grads->vector" (grads->vector grads))
               (println "grads2:" grads)
               (is (= [1] grads))
               ))))



;; TODO need to check the functions above and implement grad-check
