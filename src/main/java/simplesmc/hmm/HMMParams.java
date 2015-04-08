package simplesmc.hmm;

import java.util.Random;

import simplesmc.pmcmc.WithSignature;


/**
 * Parameters for an HMM: initial prs, transition matrix, emission matrix
 * 
 * Note: probabilities are returned in LOG
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 */
public interface HMMParams extends WithSignature
{
  public double initialLogPr(int state);
  public int sampleInitial(Random random);
  
  public double transitionLogPr(int currentState, int nextState);
  public int sampleTransition(Random random, int currentState);
  
  public double emissionLogPr(int latentState, int emission);
  public int sampleEmission(Random random, int currentState);
  
  public int nLatentStates();
  public int nObservedStates();
}
