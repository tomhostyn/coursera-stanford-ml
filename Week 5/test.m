
function [cost] = J(theta)
  cost = 2 * (theta ^4) + 2 
end


epsilon = 0.01
theta = 1


(J(theta + epsilon) - J(theta - epsilon))/(2*epsilon)