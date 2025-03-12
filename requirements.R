packages <- c(
  'IRkernel',
  'rmarkdown',
  'languageserver',
  'tidyverse',
  'ggplot2',
  'tidymodels',
  'mvtnorm'
)

install.packages(packages, repos = 'https://cloud.r-project.org/')
IRkernel::installspec(user = FALSE)