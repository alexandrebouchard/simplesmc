package simplesmc;

import java.util.Random;

import bayonet.smc.ResamplingScheme;
import briefj.opt.Option;



public class SMCOptions
{
  @Option
  public double essThreshold = 0.5;

  @Option
  public Random random = new Random(1);
  
  @Option
  public int nParticles = 100;
  
  @Option
  public ResamplingScheme resamplingScheme = ResamplingScheme.MULTINOMIAL;
}