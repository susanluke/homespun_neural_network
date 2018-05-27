(ns homespun-neural-network.logistic-regression-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [homespun-neural-network.logistic-regression :as sut]))

(defn almost=
  [n1 n2]
  (let [diff (Math/abs (- n1 n2))]
    (< diff 1e-8)))

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

(deftest coursera-tests
  (let [W [1 2]
        b 2
        X [[1 2 -1]
           [3 4 -3.2]]
        Y [1 0 1]]
    (testing "propagate test scenario on coursera"
      (let [A (sut/forward-prop X W b)
            cost (sut/cost Y A)
            grads (sut/back-prop A Y X)]
        (is (almost= 5.801545319394553 cost))
        (is (every? true?
                    (map almost= [0.99845601 2.39507239] (:dW grads))))
        (is (almost= 0.00145557813678 (:db grads)))))
    (testing "optimise scenario on coursera"
      (let [grads (sut/linear-regression X Y {:W W :b b} 0.009 100)]
        (is (every? true?
                    (map almost= [0.19033591 0.12259159] (:W grads))))
        (is (almost= 1.92535983008 (:b grads)))))))
