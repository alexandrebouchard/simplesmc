package simplesmc.pmcmc;

import blang.annotations.FactorComponent;
import blang.factors.Factor;
import simplesmc.SMCAlgorithm;



public class PMCMCFactor<P> implements Factor
{
  @FactorComponent
  public final WithSignature params;
  
  private final SMCAlgorithm<P> smcAlgorithm;

  private transient long lastSignature = -1;
  private transient double lastLogDensity = Double.NaN;
  
  @Override
  public double logDensity()
  {
    final long currentSignature = params.signature();
    
    if (currentSignature == lastSignature)
      return lastLogDensity;
    
    final double result = smcAlgorithm.sample().logNormEstimate();
    lastSignature = currentSignature;
    lastLogDensity = result;
    
    return result;
  }

  public PMCMCFactor(WithSignature params, SMCAlgorithm<P> smcAlgorithm)
  {
    this.params = params;
    this.smcAlgorithm = smcAlgorithm;
  }
}
