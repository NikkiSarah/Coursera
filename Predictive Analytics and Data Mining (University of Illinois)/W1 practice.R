library(dplyr)
library(ggplot2)
library(factoextra)

data <- read_csv('customer_segment.csv')
head(data)
summary(data)

# normalise the data 
data[, c("requests", "revenue")] = scale(data[, c("requests", "revenue")])

# use a loop to iterate over a list of cluster numbers
kmeans_cols <- data %>% select(-cust_id)

# look over 1 to n possible clusters
n_clusters = 10
wss <- numeric(n_clusters)
for(i in 1:n_clusters) {
  mod <- kmeans(kmeans_cols, centers = i)
  wss[i] <- mod$tot.withinss
}

# produce a scree plot
wss_df <- tibble(num_clusters = 1:n_clusters, wss = wss)
wss_df$diff_wss <- lag(wss_df$wss) - wss_df$wss
wss_df <- wss_df %>% gather(key = "variable", value = "value", -num_clusters)
wss_df

scree_plot <- ggplot(wss_df, aes(x = num_clusters, y = value)) +
  geom_point(size = 2) +
  geom_line(aes(colour = variable, linetype = variable)) +
  scale_colour_manual(values = c('darkred', 'steelblue')) +
  scale_x_continuous(breaks = 1:n_clusters) +
  labs(x = 'k', y = 'total within sum of squares') +
  theme_classic()
scree_plot

# generate the model with just three clusters
mod_3 <- kmeans(kmeans_cols, centers = 3)
mod_3$size

# generate a discriminant plot
fviz_cluster(mod_3, data = kmeans_cols, show.clust.cent = TRUE, ellipse.type = "norm") +
  labs(title = "Disciminant plot") + 
  theme_classic()

# The green cluster (top middle) makes an average number of service requests, and gives the highest revenue.
# The blue cluster (bottom left) makes under 20 service requests, however gives revenue mostly between 200 and 700.
# The red cluster (bottom right) makes mostly between 60 and 100 service requests, however gives the lowest revenue. This segment is the least profitable.     
