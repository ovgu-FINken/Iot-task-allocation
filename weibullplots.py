from reliability.Distributions import Weibull_Distribution as wb 
import matplotlib.pyplot as plt



dist = wb(500,4)
dist.CDF()
ax = plt.gca()
ax.set_xlabel('Time (s)')
plt.title("Weibull \n Distribution Probability Density Function \n \u03BB =500,k=4")
plt.savefig('plots/weibull_failure_CDF.png')
plt.show()

dist.PDF()
ax = plt.gca()
plt.title("Weibull \n Cumulative Distribution Function \n \u03BB =500,k=4")
ax.set_xlabel('Failure Time (s)')
plt.savefig('plots/weibull_failure_PDF.png')
plt.show()



dist = wb(120,3)
dist.CDF()
plt.savefig('plots/weibull_duration.png')
plt.show()
