generate_nested_poisson_data <- function(
    n_schools,
    n_classes_per_school,
    n_students_per_class,
    beta_0,
    beta_1,
    sigma_school,
    sigma_class
  ) {
  # Create a data frame to store the simulated data
  data <- data.frame(
    school = rep(1:n_schools, each = n_classes_per_school * n_students_per_class),
    class = rep(1:n_classes_per_school, times = n_schools * n_students_per_class),
    student = 1:(n_schools * n_classes_per_school * n_students_per_class)
  )

  # Generate random effects for schools and classes
  school_effects <- rnorm(n_schools, mean = 0, sd = sigma_school)
  class_effects <- rnorm(n_classes_per_school * n_schools, mean = 0, sd = sigma_class)

  # Assign random effects to the data
  data$school_effect <- school_effects[data$school]
  data$class_effect <- class_effects[(data$school - 1) * n_classes_per_school + data$class]

  x <- rnorm(n = nrow(data))
  # Generate the response variable y
  data$y <- rpois(nrow(data), lambda = exp(beta_0 + beta_1 * x + data$school_effect + data$class_effect))
  data$x <- x

  return(data)
}
