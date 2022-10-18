package mat4j;

import java.util.MissingResourceException;
import java.util.ResourceBundle;

/**
 * The messages class for the math library.
 *
 * @author <a href="mailto:jiangfan0576@gmail.com">Frank Jiang</a>
 * @version 1.0.0
 */
public class Messages
{
    private static final String			BUNDLE_NAME		= "com.frank.math.messages";		//$NON-NLS-1$
    private static final ResourceBundle	RESOURCE_BUNDLE	= ResourceBundle
            .getBundle(BUNDLE_NAME);

    private Messages()
    {
    }

    public static String getString(String key)
    {
        try
        {
            return RESOURCE_BUNDLE.getString(key);
        }
        catch (MissingResourceException e)
        {
            return '!' + key + '!';
        }
    }
}