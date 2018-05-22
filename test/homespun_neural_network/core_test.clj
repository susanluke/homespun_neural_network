(ns homespun-neural-network.core-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [homespun-neural-network.core :as sut]))

(deftest init-params-test
  (testing "Map is returned with the right keys"
    (is (= #{:W :b}
           (-> (sut/init-params 5) keys set)))))

(deftest forward-prop-test
  (testing "given a X, W & b, the expected values are returned"
    (let [X [[5] [7]]
          W [1 2]
          b 1]
      (is (= [(m/logistic 20.0)]
             (sut/forward-prop X W b)))))

  (testing "given a X, W & b, the expected values are returned"
    (let [X [[5 7 10] ; n-x is 2, 3 training cases
             [1 2 3]]
          W [1 2]
          b 1]
      (is (= [(m/logistic 8)
              (m/logistic 12)
              (m/logistic 17)]
             (sut/forward-prop X W b))))))

(deftest cost-test
  (testing "all activations correct gives cost 0"
    (is (> 0.01 (sut/cost [1 1 0 1]
                          [0.999 0.9999 0.001 0.999])))))

(deftest coursera-test
  (testing "test scenario on coursera"
    (let [W [1 2]
          b 2
          X [[1 2 -1]
             [3 4 -3.2]]
          Y [1 0 1]
          A (sut/forward-prop X W b)
          cost (sut/cost Y A)
          grads (sut/back-prop A Y X)]
      (println "A" A)
      (println "cost" cost)
      (println "grads" grads))))


(comment "
w = np.array([[1.],[2.]]),
b = 2.,
X = np.array([[1.,2.,-1.],
              [3.,4.,-3.2]]),
Y =  np.array([[1,0,1]])

Expected Output

cost	5.801545319394553

dw	[[ 0.99845601] [ 2.39507239]]
db	0.00145557813678


")
