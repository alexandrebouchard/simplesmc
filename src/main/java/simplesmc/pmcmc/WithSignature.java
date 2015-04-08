package simplesmc.pmcmc;


/**
 * This indicates that the object is able to summarize (hash) its 
 * internal state. 
 * 
 * This is used by PMCMC to check whether is should
 * recompute the estimate for Z, or use the old value that was 
 * already computed.
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 *
 */
public interface WithSignature
{
  public long signature();
}
