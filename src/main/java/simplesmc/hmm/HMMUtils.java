package simplesmc.hmm;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.jgrapht.UndirectedGraph;

import bayonet.distributions.Random;
import bayonet.graphs.GraphUtils;
import bayonet.marginal.DiscreteFactorGraph;
import bayonet.marginal.algo.SumProduct;

import com.google.common.collect.Iterables;


/**
 * Some utilities (static functions) for finite HMMs.
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 *
 */
public class HMMUtils
{
  /**
   * Perform forward simulation of an HMM model of the provided length
   * 
   * @return A pair, where the first item is the list of latent states, and the second item
   *   is the corresponding list of observations
   */
  public static Pair<List<Integer>,List<Integer>> generate(Random random, HMMParams params, int length)
  {
    List<Integer> 
      latents = new ArrayList<Integer>(),
      observations = new ArrayList<>();
    
    for (int iteration = 0; iteration < length; iteration++)
    {
      int currentLatent = iteration == 0 ? 
        params.sampleInitial(random) : 
        params.sampleTransition(random, Iterables.getLast(latents));
      int currentObs = params.sampleEmission(random, currentLatent);
      latents.add(currentLatent);
      observations.add(currentObs);
    }
    
    return Pair.of(latents, observations);
  }
  
  /**
   * Construct a factor graph, and use the sum product algorithm to 
   * compute the LOG probability of the data analytically
   * 
   * @return The LOG probability of the provided sequence of observations
   */
  public static double exactDataLogProbability(HMMParams parameters, List<Integer> observations)
  {
    final int len = observations.size();
    UndirectedGraph<Integer, ?> topology = GraphUtils.createChainTopology(len);
    DiscreteFactorGraph<Integer> factorGraph = new DiscreteFactorGraph<Integer>(topology);
    
    // initial distribution
    factorGraph.setUnary(0, new double[][]{initialPrs(parameters)});
    
    // transition
    for (int i = 0; i < len-1; i++)
      factorGraph.setBinary(i, i+1, transitionPrs(parameters));
    
    // observations
    for (int i = 0; i < len; i++)
    {
      int currentObs = observations.get(i);
      double [] curEmissionLikelihoods = new double[parameters.nLatentStates()];
      for (int s = 0; s < parameters.nLatentStates(); s++)
        curEmissionLikelihoods[s] = Math.exp(parameters.emissionLogPr(s, currentObs));
      factorGraph.unaryTimesEqual(i, new double[][]{curEmissionLikelihoods});
    }
    
    SumProduct<Integer> sumProd = new SumProduct<>(factorGraph);
    return sumProd.logNormalization();
  }
  
  private static double [][] transitionPrs(HMMParams parameters)
  {
    final int latentSize = parameters.nLatentStates();
    double[][] result = new double[latentSize][latentSize];
    for (int first = 0; first < latentSize; first++)
      for (int second = 0; second < latentSize; second++)
        result[first][second] = Math.exp(parameters.transitionLogPr(first, second));
    return result;
  }

  private static double[] initialPrs(HMMParams parameters)
  {
    final int latentSize = parameters.nLatentStates();
    double [] result = new double[latentSize];
    for (int latent = 0; latent < latentSize; latent++)
      result[latent] = Math.exp(parameters.initialLogPr(latent));
    return result;
  }

  private HMMUtils() {}
}
